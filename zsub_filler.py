#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import io
import re
import json
import math
import argparse
import unicodedata

VERSION = "3.0.0"

def log(msg):
    try:
        print(f"[ZFill] {msg}", file=sys.stderr, flush=True)
    except Exception:
        print("[ZFill] log error", file=sys.stderr, flush=True)

def runtime_base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

def resolve_torch_cache_dir(user_cache_dir=None):
    candidates = []
    if user_cache_dir:
        candidates.append(os.path.abspath(user_cache_dir))

    base = runtime_base_dir()
    candidates.extend([
        os.path.join(base, ".torch-cache"),
        os.path.join(base, "_internal", ".torch-cache"),
        os.path.join(base, "torch-cache"),
        os.path.join(base, "_internal", "torch-cache"),
    ])

    for c in candidates:
        if os.path.isdir(c):
            return c

    return os.path.abspath(user_cache_dir) if user_cache_dir else os.path.join(base, ".torch-cache")

def parse_srt_timestamp(ts):
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    sec, ms = s.split(".")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0

def parse_srt(srt_path):
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    entries = []

    for block in blocks:
        lines = [x.strip() for x in block.splitlines() if x.strip()]
        if len(lines) < 2:
            continue

        start_sec = None
        end_sec = None
        text_lines = []

        for line in lines:
            m = re.match(
                r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})",
                line,
            )
            if m:
                start_sec = parse_srt_timestamp(m.group(1))
                end_sec = parse_srt_timestamp(m.group(2))
            elif not re.match(r"^\d+$", line):
                text_lines.append(line)

        if start_sec is not None and end_sec is not None and text_lines:
            text = " ".join(text_lines).strip()
            if text:
                entries.append((start_sec, end_sec, text))

    return entries

def build_full_transcript(entries):
    return " ".join(x[2] for x in entries).strip()

def get_whisper_lang_from_entries(entries):
    # SRT'te dil yok; main.js isterse CLI'dan verir.
    return None

def basic_ascii_fallback(text):
    text = text.lower().replace("’", "'")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_uroman(text):
    text = text.lower().replace("’", "'")
    text = re.sub(r"([^a-z' ])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def romanize_text(text, lang_code=None):
    try:
        import uroman as ur
        u = ur.Uroman()
        romanized = u.romanize_string(text, lcode=lang_code) if lang_code else u.romanize_string(text)
        return normalize_uroman(romanized), "uroman"
    except Exception as e:
        log(f"uroman fallback: {e}")
        return basic_ascii_fallback(text), "ascii"

def split_to_samples(y, frame_size, hop_size):
    frames = []
    i = 0
    n = len(y)
    while i + frame_size <= n:
        frames.append(y[i:i + frame_size])
        i += hop_size
    return frames

def compute_rms_and_zcr(y, sr, frame_ms=20, hop_ms=10):
    frame_size = int(sr * frame_ms / 1000.0)
    hop_size = int(sr * hop_ms / 1000.0)
    frames = split_to_samples(y, frame_size, hop_size)

    rms = []
    zcr = []

    for frame in frames:
        if not frame:
            continue
        s = 0.0
        crossings = 0
        prev = frame[0]
        for idx, v in enumerate(frame):
            s += v * v
            if idx > 0 and ((prev < 0 <= v) or (prev >= 0 > v)):
                crossings += 1
            prev = v
        rms.append(math.sqrt(s / len(frame)))
        zcr.append(crossings / max(1, len(frame) - 1))

    return rms, zcr, hop_size

def get_window_stats(values, start_sec, end_sec, hop_size, sr):
    if not values:
        return 0.0, 0.0
    hop_sec = hop_size / float(sr)
    s = max(0, int(start_sec / hop_sec))
    e = min(len(values), int(end_sec / hop_sec) + 1)
    seg = values[s:e]
    if not seg:
        return 0.0, 0.0
    mean = sum(seg) / len(seg)
    var = sum((x - mean) ** 2 for x in seg) / len(seg)
    return mean, math.sqrt(var)

def detect_fillers(audio_path, srt_path, out_path, cache_dir=None,
                   min_gap=0.12, max_gap=1.00, energy_margin=0.006,
                   zcr_min=0.01, zcr_max=0.22, debug=False, lang_code=None):
    import torch
    import torchaudio
    from torchaudio.pipelines import MMS_FA

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio yok: {audio_path}")
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT yok: {srt_path}")

    entries = parse_srt(srt_path)
    if not entries:
        raise RuntimeError("SRT okunamadi veya transcript bos")

    transcript_raw = build_full_transcript(entries)
    transcript_norm, norm_mode = romanize_text(transcript_raw, lang_code=lang_code)

    if not transcript_norm:
        raise RuntimeError("Transcript normalize edilemedi")

    cache_root = resolve_torch_cache_dir(cache_dir)
    os.makedirs(cache_root, exist_ok=True)
    torch.hub.set_dir(cache_root)

    checkpoints_dir = os.path.join(cache_root, "hub", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Cihaz: {device}")
    log(f"Normalize: {norm_mode}")
    if debug:
        log(f"Transcript raw: {transcript_raw[:200]}")
        log(f"Transcript norm: {transcript_norm[:200]}")

    bundle = MMS_FA
    model = bundle.get_model(with_star=False, dl_kwargs={"model_dir": checkpoints_dir}).to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    sample_rate = bundle.sample_rate

    waveform, sr = torchaudio.load(audio_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        sr = sample_rate

    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
    emission = emission[0].cpu()

    token_seqs = tokenizer([transcript_norm])
    span_seqs = aligner(emission, token_seqs)
    token_spans = span_seqs[0] if span_seqs else []

    if not token_spans:
        raise RuntimeError("Forced alignment span uretemedi")

    total_frames = emission.size(0)
    duration_sec = waveform.size(1) / float(sr)
    frame_sec = duration_sec / max(1, total_frames)

    y = waveform[0].cpu().numpy().tolist()
    rms_vals, zcr_vals, hop_size = compute_rms_and_zcr(y, sr, frame_ms=20, hop_ms=10)

    noise_floor = 0.003
    if rms_vals:
        sorted_rms = sorted(rms_vals)
        noise_floor = sorted_rms[max(0, int(len(sorted_rms) * 0.15))]

    def span_start_sec(span):
        return float(span.start) * frame_sec

    def span_end_sec(span):
        return float(span.end) * frame_sec

    cuts = []

    prev_end = 0.0
    for idx, span in enumerate(token_spans):
        cur_start = span_start_sec(span)
        gap_start = prev_end
        gap_end = cur_start
        gap_dur = gap_end - gap_start

        if min_gap <= gap_dur <= max_gap:
            avg_rms, std_rms = get_window_stats(rms_vals, gap_start, gap_end, hop_size, sr)
            avg_zcr, std_zcr = get_window_stats(zcr_vals, gap_start, gap_end, hop_size, sr)

            is_voicedish = zcr_min <= avg_zcr <= zcr_max
            is_energetic = avg_rms > (noise_floor + energy_margin)

            if is_energetic:
                pad = min(0.03, gap_dur * 0.08)
                cuts.append({
                    "start": round(gap_start + pad, 3),
                    "end": round(gap_end - pad, 3),
                    "dur": round(max(0.0, gap_dur - pad * 2), 3),
                    "reason": "filler:mms_fa_blank" if is_voicedish else "breath:mms_fa_blank",
                    "avg_rms": round(avg_rms, 5),
                    "avg_zcr": round(avg_zcr, 5),
                })
                if debug:
                    log(
                        f"GAP {gap_start:.3f}-{gap_end:.3f} "
                        f"dur:{gap_dur:.3f} rms:{avg_rms:.5f} zcr:{avg_zcr:.5f} "
                        f"-> {cuts[-1]['reason']}"
                    )

        prev_end = span_end_sec(span)

    tail_gap = duration_sec - prev_end
    if min_gap <= tail_gap <= max_gap:
        avg_rms, std_rms = get_window_stats(rms_vals, prev_end, duration_sec, hop_size, sr)
        avg_zcr, std_zcr = get_window_stats(zcr_vals, prev_end, duration_sec, hop_size, sr)
        is_voicedish = zcr_min <= avg_zcr <= zcr_max
        is_energetic = avg_rms > (noise_floor + energy_margin)
        if is_energetic:
            pad = min(0.03, tail_gap * 0.08)
            cuts.append({
                "start": round(prev_end + pad, 3),
                "end": round(duration_sec - pad, 3),
                "dur": round(max(0.0, tail_gap - pad * 2), 3),
                "reason": "filler:mms_fa_blank" if is_voicedish else "breath:mms_fa_blank",
                "avg_rms": round(avg_rms, 5),
                "avg_zcr": round(avg_zcr, 5),
            })

    cuts = [c for c in cuts if c["end"] - c["start"] >= 0.08]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cuts, f, ensure_ascii=False, indent=2)

    log(f"Kaydedildi: {out_path}")
    log(f"Toplam kesim: {len(cuts)}")
    return cuts

def main():
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    pa = argparse.ArgumentParser(description="ZSub Filler Engine v3.0 - MMS_FA Multilingual")
    pa.add_argument("--audio", required=True)
    pa.add_argument("--srt", required=True)
    pa.add_argument("--out", required=True)
    pa.add_argument("--cache-dir", default="")
    pa.add_argument("--lang", default="")
    pa.add_argument("--min-gap", type=float, default=0.12)
    pa.add_argument("--max-gap", type=float, default=1.00)
    pa.add_argument("--energy-margin", type=float, default=0.006)
    pa.add_argument("--zcr-min", type=float, default=0.01)
    pa.add_argument("--zcr-max", type=float, default=0.22)
    pa.add_argument("--debug", action="store_true")
    a = pa.parse_args()

    cuts = detect_fillers(
        audio_path=a.audio,
        srt_path=a.srt,
        out_path=a.out,
        cache_dir=a.cache_dir or None,
        min_gap=a.min_gap,
        max_gap=a.max_gap,
        energy_margin=a.energy_margin,
        zcr_min=a.zcr_min,
        zcr_max=a.zcr_max,
        debug=a.debug,
        lang_code=a.lang or None,
    )

    log(f"TAMAM | {len(cuts)} kesim")

if __name__ == "__main__":
    main()
