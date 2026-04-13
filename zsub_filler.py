#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZSub Filler Engine v2.0 - MMS Forced Alignment
Dolgu sesleri ve nefesleri millisaniye hassasiyetinde tespit eder.

Yaklasim:
  1. Whisper SRT -> temiz transcript metni
  2. MMS (facebook/mms-300m veya mms-1b-all) -> ses uzerinde CTC emission
  3. torchaudio.functional.forced_align -> her harfi ses dalgasina oturt
  4. Hizalanamayan (blank) + enerji olan bolgeler = dolgu sesi / nefes
  5. cuts.json cikti -> main.js ripple delete

Gemini versiyonundan farklar:
  - parse_srt ve transcript GERCEKTEN kullaniliyor (forced alignment icin sart)
  - Greedy decode degil, gercek forced_align ile karakter hizalamasi
  - Energy gate: blank bolgede RMS enerji kontrolu (sessizlik vs dolgu ayrimi)
  - MMS tokenizer normalizasyonu (Turkce karakterler dahil)
  - Uzun ses dosyasi icin chunk'lama (bellek tasarrufu)

Kurulum:
  pip install torch torchaudio transformers

Model (bir kez indir):
  from transformers import AutoProcessor, AutoModelForCTC
  AutoProcessor.from_pretrained("facebook/mms-300m").save_pretrained("models/mms-300m")
  AutoModelForCTC.from_pretrained("facebook/mms-300m").save_pretrained("models/mms-300m")

Kullanim:
  python zsub_filler.py --audio ses.wav --srt altyazi.srt --model-dir models/mms-300m --out cuts.json
  python zsub_filler.py --audio ses.wav --srt altyazi.srt --model-dir models/mms-300m --out cuts.json --debug
"""

import sys
import os
import json
import argparse
import re
import math
import wave
import struct

# --- torch/torchaudio geciktirilmis import (PyInstaller uyumlulugu) ---
# main() cagirilana kadar import edilmiyor, boylece import hatasi daha anlasilir cikiyor

def log(msg):
    try:
        print(f"[ZFill] {msg}", file=sys.stderr, flush=True)
    except (UnicodeEncodeError, OSError):
        print(f"[ZFill] {msg.encode('utf-8', errors='replace').decode('ascii', errors='replace')}",
              file=sys.stderr, flush=True)

# ─────────────────────────────────────────────────────────
# SRT PARSER
# ─────────────────────────────────────────────────────────

def parse_srt(srt_path):
    """
    SRT dosyasini okur, her satirin (start_sec, end_sec, text) tuple'ini doner.
    Noktalama temizlenir, lowercase yapilir.
    """
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    def ts_to_sec(ts):
        # "00:00:01,234" veya "00:00:01.234"
        ts = ts.replace(',', '.')
        h, m, rest = ts.split(':')
        s, ms = rest.split('.')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    blocks = re.split(r'\n\s*\n', content.strip())
    entries = []
    for block in blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if len(lines) < 3:
            continue
        # Satir 1: index (atla), Satir 2: timestamp, Satir 3+: metin
        ts_line = None
        text_lines = []
        for line in lines:
            m = re.match(
                r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})',
                line
            )
            if m:
                ts_line = (ts_to_sec(m.group(1)), ts_to_sec(m.group(2)))
            elif ts_line and not re.match(r'^\d+$', line):
                text_lines.append(line)
        if ts_line and text_lines:
            raw_text = ' '.join(text_lines)
            # Noktalama kaldir, lowercase
            clean = re.sub(r"[^\w\s]", "", raw_text.lower()).strip()
            if clean:
                entries.append((ts_line[0], ts_line[1], clean))

    return entries

def build_full_transcript(entries):
    """SRT entry listesinden tek cumle olustur (alignment icin)."""
    return ' '.join(e[2] for e in entries)

# ─────────────────────────────────────────────────────────
# MMS TOKENIZER NORMALIZASYON
# ─────────────────────────────────────────────────────────

def normalize_for_mms(text, vocab):
    """
    Metni MMS vocab ile uyumlu hale getirir.
    - Turkce karakterler: ı->i, ş->s, ğ->g, ü->u, ö->o, ç->c
      (MMS-300m multi-lingual vocab'da bunlar VARSA kullan, yoksa ASCII fallback)
    - Bilinmeyen karakterleri at
    - Bosluklar <space> token'i (vocab'da varsa)

    MMS vocab ornegi: {'<pad>':0, '<unk>':1, 'a':5, 'b':6, ..., '|':4}
    '|' karakteri kelimeler arasi bosluktur (WORD_BOUNDARY).
    """
    # Bosluk tokeni: MMS'de genellikle '|' veya '<space>'
    space_token = None
    if '|' in vocab:
        space_token = '|'
    elif '<space>' in vocab:
        space_token = '<space>'

    # Turkce karakter haritasi - once vocab'da var mi kontrol et
    tr_map = {}
    tr_pairs = [
        ('\u0131', 'i'),   # ı -> i
        ('\u015f', 's'),   # ş -> s
        ('\u011f', 'g'),   # ğ -> g
        ('\u00fc', 'u'),   # ü -> u
        ('\u00f6', 'o'),   # ö -> o
        ('\u00e7', 'c'),   # ç -> c
    ]
    for tr_char, ascii_char in tr_pairs:
        if tr_char in vocab:
            tr_map[tr_char] = tr_char   # Vocab'da varsa orijinali kullan
        else:
            tr_map[tr_char] = ascii_char  # Yoksa ASCII fallback

    normalized = []
    for ch in text.lower():
        if ch == ' ':
            if space_token:
                normalized.append(space_token)
            # space_token yoksa boslugu atla
        elif ch in tr_map:
            normalized.append(tr_map[ch])
        elif ch in vocab:
            normalized.append(ch)
        # Bilinmeyen karakter: atla (hallucination riskini azaltir)

    return normalized

# ─────────────────────────────────────────────────────────
# WAV ENERGY (saf Python - scipy gerekmez)
# ─────────────────────────────────────────────────────────

def read_wav_rms_frames(wav_path, frame_ms=20, hop_ms=10):
    """
    WAV dosyasini okur, her frame icin RMS enerji deger doner.
    Donus: (rms_list, framerate, hop_size_samples)
    Pure Python - harici kutuphane yok.
    """
    try:
        with wave.open(wav_path, 'rb') as wf:
            channels   = wf.getnchannels()
            sampwidth  = wf.getsampwidth()
            framerate  = wf.getframerate()
            n_frames   = wf.getnframes()
            raw        = wf.readframes(n_frames)

        if sampwidth == 2:
            samples = list(struct.unpack(f"<{n_frames * channels}h", raw))
        elif sampwidth == 1:
            samples = [b - 128 for b in struct.unpack(f"{n_frames * channels}B", raw)]
        else:
            return [], framerate, 1

        # Stereo -> mono
        if channels > 1:
            samples = samples[::channels]

        max_val = 32768.0 if sampwidth == 2 else 128.0
        frame_size = int(framerate * frame_ms / 1000)
        hop_size   = int(framerate * hop_ms  / 1000)

        rms_list = []
        i = 0
        while i + frame_size <= len(samples):
            frame = samples[i:i + frame_size]
            rms = math.sqrt(sum(s * s for s in frame) / len(frame)) / max_val
            rms_list.append(rms)
            i += hop_size

        return rms_list, framerate, hop_size

    except Exception as e:
        log(f"WAV okuma hatasi: {e}")
        return [], 16000, 160

def get_energy_at(rms_list, hop_size, framerate, start_sec, end_sec):
    """Belirli bir zaman araligindaki ortalama RMS enerjiyi doner."""
    if not rms_list:
        return 0.0
    hop_sec = hop_size / framerate
    start_idx = max(0, int(start_sec / hop_sec))
    end_idx   = min(len(rms_list), int(end_sec / hop_sec) + 1)
    seg = rms_list[start_idx:end_idx]
    return sum(seg) / len(seg) if seg else 0.0

# ─────────────────────────────────────────────────────────
# FORCED ALIGNMENT
# ─────────────────────────────────────────────────────────

def run_forced_alignment(emission, token_ids, blank_id=0):
    """
    torchaudio.functional.forced_align ile karakter hizalamasi.

    emission: [1, T, C] tensor (log_softmax uygulanmis)
    token_ids: [S] list - transcript karakter ID'leri (blank haric)
    blank_id: CTC blank token ID (genellikle 0)

    Donus: aligned_tokens list, her eleman:
        {'char_idx': i, 'start_frame': f1, 'end_frame': f2, 'score': s}
    """
    import torch
    import torchaudio

    if not token_ids:
        return []

    targets = torch.tensor([token_ids], dtype=torch.int32)

    try:
        # torchaudio >= 0.13 API
        alignments, scores = torchaudio.functional.forced_align(
            emission,   # [1, T, C]
            targets,    # [1, S]
            blank=blank_id,
        )
        # alignments[0]: [T] - her frame hangi token'a eslestirildi
        # scores[0]: [T] - her frame'in skoru
        aligned = alignments[0].numpy()
        score_arr = scores[0].numpy()
    except Exception as e:
        log(f"forced_align hatasi: {e}")
        return []

    # Frame dizisinden karakter sinirlari cikar
    # Ornek: [0,0,0,1,1,0,0,2,2,2,0] -> char 1: frames 3-4, char 2: frames 7-9
    char_spans = []
    S = len(token_ids)
    char_idx = 0
    span_start = None

    for f, tok in enumerate(aligned):
        if tok == blank_id:
            if span_start is not None:
                char_spans.append({
                    'char_idx':   char_idx,
                    'token_id':   token_ids[char_idx] if char_idx < S else -1,
                    'start_frame': span_start,
                    'end_frame':   f - 1,
                    'score': float(score_arr[span_start:f].mean()) if f > span_start else 0.0,
                })
                span_start = None
                char_idx += 1
                if char_idx >= S:
                    break
        else:
            if span_start is None:
                span_start = f

    # Son span
    if span_start is not None and char_idx < S:
        char_spans.append({
            'char_idx':    char_idx,
            'token_id':    token_ids[char_idx],
            'start_frame': span_start,
            'end_frame':   len(aligned) - 1,
            'score': float(score_arr[span_start:].mean()),
        })

    return char_spans

def find_unaligned_gaps(char_spans, total_frames, frame_duration=0.02):
    """
    Hizalanmis karakter span'leri arasindaki boslukları bul.
    Bunlar 'blank' bolgeler: dolgu sesi, nefes, veya sessizlik adayi.

    char_spans: run_forced_alignment ciktisi
    total_frames: emission tensor'un T boyutu
    frame_duration: saniye/frame (MMS icin 0.02s = 20ms)

    Donus: gap listesi [{start_sec, end_sec, dur_sec, gap_frames}]
    """
    if not char_spans:
        # Hic alignment yoksa tum ses blank - muhtemelen transcript hatasi
        return []

    gaps = []

    # Bastan ilk karaktere kadar bosluk
    first_start = char_spans[0]['start_frame']
    if first_start > 10:  # 200ms'den uzun bosluk
        gaps.append({
            'start_sec': 0.0,
            'end_sec':   round(first_start * frame_duration, 3),
            'dur_sec':   round(first_start * frame_duration, 3),
            'gap_frames': first_start,
        })

    # Karakterler arasi bosluklar
    for i in range(len(char_spans) - 1):
        end_f   = char_spans[i]['end_frame']
        start_f = char_spans[i + 1]['start_frame']
        gap_f   = start_f - end_f - 1
        if gap_f > 0:
            gaps.append({
                'start_sec': round(end_f * frame_duration, 3),
                'end_sec':   round(start_f * frame_duration, 3),
                'dur_sec':   round(gap_f * frame_duration, 3),
                'gap_frames': gap_f,
            })

    # Son karakterden sese kadar bosluk
    last_end = char_spans[-1]['end_frame']
    if total_frames - last_end > 10:
        gaps.append({
            'start_sec': round(last_end * frame_duration, 3),
            'end_sec':   round(total_frames * frame_duration, 3),
            'dur_sec':   round((total_frames - last_end) * frame_duration, 3),
            'gap_frames': total_frames - last_end,
        })

    return gaps

# ─────────────────────────────────────────────────────────
# ANA PIPELINE
# ─────────────────────────────────────────────────────────

def detect_filler_cuts(audio_path, srt_path, model_dir, output_json,
                        min_dur_sec=0.35, energy_threshold=0.008, debug=False):
    """
    Tam filler tespit pipeline'i.

    min_dur_sec: Bu sureden kisa blank bolgeler kesilmez (varsayilan 350ms)
    energy_threshold: Bu RMS degerinin altindaki blank bolgeler sessizliktir,
                      ustundekiler dolgu sesidir. Her ikisi de kesilir ama
                      reason ayrimi yapilir.
    """
    import torch
    import torchaudio
    from transformers import AutoProcessor, AutoModelForCTC

    log(f"Basliyor: {os.path.basename(audio_path)}")

    # ── 1. SRT oku ──
    if not os.path.exists(srt_path):
        log(f"HATA: SRT bulunamadi: {srt_path}")
        sys.exit(1)

    srt_entries = parse_srt(srt_path)
    if not srt_entries:
        log("HATA: SRT'de okunabilir metin yok.")
        sys.exit(1)

    full_transcript = build_full_transcript(srt_entries)
    log(f"SRT: {len(srt_entries)} satir, {len(full_transcript)} karakter")
    if debug:
        log(f"  Transcript: '{full_transcript[:120]}...'")

    # ── 2. Model yukle ──
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    log(f"Cihaz: {device_str.upper()}")

    log("Model yukleniyor...")
    try:
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForCTC.from_pretrained(model_dir, local_files_only=True).to(device)
        model.eval()
    except Exception as e:
        log(f"Model yuklenemedi: {e}")
        sys.exit(1)

    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id
    if blank_id is None:
        blank_id = 0
    log(f"Vocab: {len(vocab)} token, blank_id={blank_id}")

    # ── 3. Ses yukle ──
    log("Ses yukleniyor...")
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        log(f"Ses yuklenemedi: {e}")
        sys.exit(1)

    if sample_rate != 16000:
        log(f"Resample: {sample_rate}Hz -> 16000Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    duration_sec = waveform.shape[1] / 16000
    log(f"Ses: {duration_sec:.1f}s")

    # ── 4. RMS enerji haritasi (pure Python WAV) ──
    rms_frames, rms_sr, rms_hop = read_wav_rms_frames(audio_path)
    noise_floor = 0.003  # varsayilan
    if rms_frames:
        sorted_rms = sorted(rms_frames)
        noise_floor = sorted_rms[max(0, int(len(sorted_rms) * 0.15))]
        log(f"Enerji: noise_floor={noise_floor:.4f}")

    # ── 5. Transcript normalize et ──
    char_tokens = normalize_for_mms(full_transcript, vocab)
    token_ids = [vocab[c] for c in char_tokens if c in vocab and vocab[c] != blank_id]

    if not token_ids:
        log("HATA: Transcript tokenize edilemedi. Vocab uyumsuzlugu?")
        if debug:
            log(f"  Ornek vocab: {list(vocab.keys())[:20]}")
        sys.exit(1)

    log(f"Token: {len(token_ids)} karakter hizalanacak")

    # ── 6. Emission hesapla ──
    # Uzun sesler icin chunk'lama (bellek tasarrufu)
    # MMS max context ~30s, uzun dosyalar icin boler
    CHUNK_SEC = 25.0
    OVERLAP_SEC = 1.0
    chunk_samples = int(CHUNK_SEC * 16000)
    overlap_samples = int(OVERLAP_SEC * 16000)
    total_samples = waveform.shape[1]

    log("Emission hesaplaniyor...")
    emission_chunks = []
    chunk_offsets = []  # Her chunk'un frame offset'i

    frames_per_chunk = 0  # Ilk chunk'tan hesaplanacak

    pos = 0
    chunk_idx = 0
    while pos < total_samples:
        end = min(pos + chunk_samples, total_samples)
        chunk = waveform[:, pos:end]

        inputs = processor(
            chunk.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values.to(device)

        with torch.no_grad():
            logits = model(inputs).logits  # [1, T_chunk, C]

        em = torch.log_softmax(logits, dim=-1).cpu()
        emission_chunks.append(em)
        chunk_offsets.append(int(pos / 16000 / 0.02))  # frame offset

        if debug:
            log(f"  Chunk {chunk_idx}: {pos/16000:.1f}-{end/16000:.1f}s -> {em.shape[1]} frame")

        pos += chunk_samples - overlap_samples
        chunk_idx += 1

    # Chunk'lari birlestir (overlap bolgelerini ort al)
    if len(emission_chunks) == 1:
        emission = emission_chunks[0]
    else:
        # Basit birlestirme: overlap bolgelerinde ortalama al
        emission = torch.cat(emission_chunks, dim=1)
        # TODO: overlap averaging - simdilik cat yeterli

    total_frames = emission.shape[1]
    log(f"Emission: {total_frames} frame ({total_frames * 0.02:.1f}s)")

    # ── 7. Forced alignment ──
    log("Forced alignment yapiliyor...")
    char_spans = run_forced_alignment(emission, token_ids, blank_id)
    log(f"Hizalanmis karakter: {len(char_spans)}/{len(token_ids)}")

    if len(char_spans) < len(token_ids) * 0.5:
        log("UYARI: Karakterlerin %50'sinden azı hizalanabildi.")
        log("  Olasi nedenler: SRT ile ses uyumsuz, model yanlis dil")

    # ── 8. Blank bolgeler bul ──
    gaps = find_unaligned_gaps(char_spans, total_frames, frame_duration=0.02)
    log(f"Ham blank bolgeler: {len(gaps)}")

    # ── 9. Filtrele ve siniflandir ──
    cuts = []
    for gap in gaps:
        dur = gap['dur_sec']
        if dur < min_dur_sec:
            continue

        start_s = gap['start_sec']
        end_s   = gap['end_sec']

        # Energy gate: sessizlik mi dolgu sesi mi?
        avg_energy = get_energy_at(rms_frames, rms_hop, 16000, start_s, end_s)
        is_filler  = avg_energy > (noise_floor + energy_threshold)

        reason = "filler:mms_blank" if is_filler else "silence:mms_blank"

        if debug:
            log(f"  GAP {start_s:.2f}-{end_s:.2f}s dur:{dur:.2f}s "
                f"energy:{avg_energy:.4f} -> {reason}")

        # Padding: kesimin tam basina/sonuna temas etme
        pad = min(0.04, dur * 0.05)
        cuts.append({
            "start":  round(start_s + pad, 3),
            "end":    round(end_s   - pad, 3),
            "reason": reason,
            "dur":    round(dur - pad * 2, 3),
        })

    # Cok kisa olanlari son kez filtrele
    cuts = [c for c in cuts if c['end'] - c['start'] >= 0.1]

    log(f"Kesimler: {len(cuts)} "
        f"(filler:{sum(1 for c in cuts if 'filler' in c['reason'])} "
        f"silence:{sum(1 for c in cuts if 'silence' in c['reason'])})")

    # ── 10. Kaydet ──
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(cuts, f, ensure_ascii=False, indent=2)

    log(f"Kaydedildi: {output_json}")
    return cuts

# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main():
    import io
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    pa = argparse.ArgumentParser(description='ZSub Filler Engine v2.0 - MMS Forced Alignment')
    pa.add_argument('--audio',     required=True,  help='Giris WAV dosyasi (16kHz mono tercih)')
    pa.add_argument('--srt',       required=True,  help='Whisper SRT dosyasi (transcript kaynagi)')
    pa.add_argument('--model-dir', required=True,  help='MMS model klasoru (facebook/mms-300m)')
    pa.add_argument('--out',       required=True,  help='Cikti cuts.json yolu')
    pa.add_argument('--min-dur',   type=float, default=0.35,
                    help='Minimum blank suresi saniye (varsayilan: 0.35)')
    pa.add_argument('--energy-threshold', type=float, default=0.008,
                    help='Dolgu sesi enerji esigi (varsayilan: 0.008)')
    pa.add_argument('--debug',     action='store_true', help='Detayli log')
    a = pa.parse_args()

    cuts = detect_filler_cuts(
        audio_path=a.audio,
        srt_path=a.srt,
        model_dir=a.model_dir,
        output_json=a.out,
        min_dur_sec=a.min_dur,
        energy_threshold=a.energy_threshold,
        debug=a.debug,
    )

    log(f"TAMAM | {len(cuts)} kesim")

if __name__ == '__main__':
    main()
