#!/usr/bin/env python3
"""
ZSub Transcription Engine
Kelime bazlı timestamp, VAD, hallüsinasyon önleme, CUDA/CPU otomatik
"""

import sys
import os
import argparse
import json
import re

def parse_args():
    parser = argparse.ArgumentParser(description='ZSub Transcription Engine')
    parser.add_argument('-m', '--model', required=True, help='Model klasörü')
    parser.add_argument('-f', '--file', required=True, help='Ses dosyası')
    parser.add_argument('-l', '--language', default='tr')
    parser.add_argument('--words-per-line', type=int, default=4)
    parser.add_argument('-of', '--output', required=True, help='Çıktı baz adı')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--compute-type', default='auto')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Sadece cuts.json üret, SRT oluşturma')
    return parser.parse_args()

def seconds_to_srt(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = min(999, int(round((s % 1) * 1000)))
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

# Bilinen hallüsinasyon kalıpları
HALLUCINATION_PATTERNS = [
    r'^\s*$',
    r'teşekkür ederim\.?\s*$',
    r'teşekkürler\.?\s*$',
    r'altyazı\s',
    r'subtitle',
    r'transcribed by',
    r'www\.',
    r'\.com',
    r'copyright',
    r'subtitles?\s+by',
    r'kanal[ıi]\s+(beğen|abone)',
    r'beğen(in|meyi unutmayın)',
    r'abone ol',
]

def is_hallucination(text):
    text_lower = text.lower().strip()
    if not text_lower:
        return True
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False

def detect_device(device_arg, compute_arg):
    if device_arg == 'auto':
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
    else:
        device = device_arg

    if compute_arg == 'auto':
        compute_type = 'int8_float16' if device == 'cuda' else 'int8'
    else:
        compute_type = compute_arg

    return device, compute_type

def build_srt(all_words, words_per_line):
    if not all_words:
        return ""

    subs = []
    i = 0
    while i < len(all_words):
        group = all_words[i:i + words_per_line]
        text  = ' '.join(w['word'].strip() for w in group).strip()
        if not text:
            i += words_per_line
            continue

        start = group[0]['start']
        end   = group[-1]['end']
        if end <= start:
            end = start + 0.1

        subs.append({'start': start, 'end': end, 'text': text})
        i += words_per_line

    # Gap uygula
    MIN_GAP = 0.08
    for j in range(len(subs) - 1):
        if subs[j]['end'] > subs[j+1]['start'] - MIN_GAP:
            subs[j]['end'] = max(subs[j]['start'] + 0.05, subs[j+1]['start'] - MIN_GAP)

    lines = []
    for idx, sub in enumerate(subs, 1):
        lines.append(str(idx))
        lines.append(f"{seconds_to_srt(sub['start'])} --> {seconds_to_srt(sub['end'])}")
        lines.append(sub['text'])
        lines.append('')

    return '\n'.join(lines)

def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"[ZSub] HATA: Model bulunamadı: {args.model}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.file):
        print(f"[ZSub] HATA: Ses dosyası bulunamadı: {args.file}", file=sys.stderr)
        sys.exit(1)

    device, compute_type = detect_device(args.device, args.compute_type)
    print(f"[ZSub] Device: {device} | Compute: {compute_type}", file=sys.stderr)
    print(f"[ZSub] Model: {args.model}", file=sys.stderr)

    try:
        from faster_whisper import WhisperModel

        # VRAM miktarına göre dinamik ayarlar
        # Az VRAM → güvenli/yavaş, Çok VRAM → hızlı/agresif
        vram_gb = 0
        if device == 'cuda':
            try:
                import torch
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"[ZSub] VRAM: {vram_gb:.1f}GB", file=sys.stderr)
            except:
                vram_gb = 6  # Bilinmiyorsa güvenli varsayılan

        if device == 'cuda':
            if vram_gb >= 12:
                # 12GB+ (RTX 5060 Ti 16GB, RTX 3080 vb.) — tam hız
                beam_size   = 5
                batch_size  = 16
                chunk_len   = 30
                num_workers = 2
            elif vram_gb >= 8:
                # 8-12GB (RTX 3070, RTX 4060 Ti vb.) — dengeli
                beam_size   = 5
                batch_size  = 8
                chunk_len   = 30
                num_workers = 1
            else:
                # 4-8GB (RTX 2060, RTX 3060 vb.) — güvenli
                beam_size   = 3
                batch_size  = 4
                chunk_len   = 25
                num_workers = 1
        else:
            # CPU
            beam_size   = 3
            batch_size  = 1
            chunk_len   = 30
            num_workers = 1

        cpu_threads = min(4, os.cpu_count() or 4)

        model = WhisperModel(
            args.model,
            device=device,
            compute_type=compute_type,
            local_files_only=True,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )

        print(f"[ZSub] Model yuklendi | device:{device} vram:{vram_gb:.1f}GB beam:{beam_size} batch:{batch_size}", file=sys.stderr)

        # CUDA: BatchedInferencePipeline ile 2-4x hız artışı
        # CPU: normal transcribe
        if device == 'cuda':
            try:
                from faster_whisper import BatchedInferencePipeline
                pipeline = BatchedInferencePipeline(model=model)
                segments, info = pipeline.transcribe(
                    args.file,
                    language=args.language if args.language != 'auto' else None,
                    word_timestamps=True,
                    batch_size=batch_size,
                    vad_filter=True,
                    vad_parameters=dict(
                        threshold=0.45,
                        min_speech_duration_ms=250,
                        max_speech_duration_s=float(chunk_len),
                        min_silence_duration_ms=500,
                        speech_pad_ms=400,
                    ),
                )
                print(f"[ZSub] Batched mod aktif", file=sys.stderr)
            except Exception as e:
                print(f"[ZSub] Batched mod basarisiz, normal mod: {e}", file=sys.stderr)
                segments, info = model.transcribe(
                    args.file,
                    language=args.language if args.language != 'auto' else None,
                    word_timestamps=True,
                    beam_size=beam_size,
                    vad_filter=True,
                    vad_parameters=dict(
                        threshold=0.45,
                        min_speech_duration_ms=250,
                        max_speech_duration_s=float(chunk_len),
                        min_silence_duration_ms=500,
                        speech_pad_ms=400,
                    ),
                    no_speech_threshold=0.6,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    condition_on_previous_text=True,
                    temperature=0.0,
                    chunk_length=chunk_len,
                )
        else:
            segments, info = model.transcribe(
                args.file,
                language=args.language if args.language != 'auto' else None,
                word_timestamps=True,
                beam_size=beam_size,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.45,
                    min_speech_duration_ms=250,
                    max_speech_duration_s=float(chunk_len),
                    min_silence_duration_ms=500,
                    speech_pad_ms=400,
                ),
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                condition_on_previous_text=True,
                temperature=0.0,
                chunk_length=chunk_len,
            )

        # Türkçe dolgu sesleri — üçü birden sağlanırsa kesilecek
        FILLER_WORDS = {
            'ıı', 'ıı.', 'eee', 'ee', 'mmm', 'mm', 'hmm', 'hm', 'aha',
            'şey', 'yani', 'işte', 'hani', 'ya', 'ha', 'he', 'ih',
            'uhh', 'umm', 'uh', 'um', 'ah', 'oh'
        }
        FILLER_PROB_THRESHOLD  = 0.6   # güven skoru bu altındaysa şüpheli
        FILLER_DUR_THRESHOLD   = 0.4   # 400ms'den kısa kelimeler şüpheli
        SILENCE_GAP_THRESHOLD  = 0.5   # 500ms üzeri boşluklar "sessizlik"
        MIN_CUT_DURATION       = 0.15  # 150ms'den kısa kesim yapma

        all_words  = []
        cut_ranges = []  # Kesilecek aralıklar
        skipped    = 0
        prev_end   = 0.0

        for segment in segments:
            if is_hallucination(segment.text):
                skipped += 1
                print(f"[ZSub] Hallucinasyon atlandi: {segment.text.strip()[:50]}", file=sys.stderr)
                continue

            if hasattr(segment, 'no_speech_prob') and segment.no_speech_prob > 0.6:
                skipped += 1
                continue

            if segment.words:
                for word in segment.words:
                    w = word.word.strip().lower().rstrip('.,!?')
                    if not w:
                        continue

                    dur  = word.end - word.start
                    prob = getattr(word, 'probability', 1.0)

                    # Sessizlik boşluğu tespiti
                    gap = word.start - prev_end
                    if prev_end > 0 and gap >= SILENCE_GAP_THRESHOLD:
                        cut_ranges.append({
                            'start':  round(prev_end, 3),
                            'end':    round(word.start, 3),
                            'reason': 'silence',
                            'dur':    round(gap, 3)
                        })

                    # Dolgu sesi tespiti — üç kriter birden
                    is_filler = (
                        w in FILLER_WORDS and
                        dur < FILLER_DUR_THRESHOLD and
                        prob < FILLER_PROB_THRESHOLD
                    )

                    if is_filler:
                        cut_ranges.append({
                            'start':  round(word.start, 3),
                            'end':    round(word.end, 3),
                            'reason': f'filler:{w}',
                            'dur':    round(dur, 3)
                        })
                        prev_end = word.end
                        continue  # SRT'ye ekleme

                    all_words.append({
                        'word':  word.word,
                        'start': word.start,
                        'end':   word.end,
                    })
                    prev_end = word.end

        # Çok kısa kesimler ve tekrarları temizle
        cut_ranges = [c for c in cut_ranges if c['dur'] >= MIN_CUT_DURATION]

        print(f"[ZSub] {len(all_words)} kelime | {skipped} segment atlandi | {len(cut_ranges)} kesim tespit edildi", file=sys.stderr)

        # Kesim listesini JSON olarak kaydet (her modda)
        cuts_path = args.output + '.cuts.json'
        with open(cuts_path, 'w', encoding='utf-8') as f:
            json.dump(cut_ranges, f, ensure_ascii=False, indent=2)
        print(f"[ZSub] Kesim listesi: {cuts_path} ({len(cut_ranges)} adet)", file=sys.stderr)

        # Analyze-only modunda SRT oluşturma
        if args.analyze_only:
            print(f"[ZSub] Analiz modu — SRT olusturulmadi", file=sys.stderr)
            print(f"[ZSub] TAMAM", file=sys.stderr)
            return

        if not all_words:
            print("[ZSub] HATA: Hic kelime bulunamadi", file=sys.stderr)
            sys.exit(1)

        srt_content = build_srt(all_words, args.words_per_line)
        output_path = args.output + '.srt'

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        print(f"[ZSub] SRT kaydedildi: {output_path}", file=sys.stderr)
        print(f"[ZSub] TAMAM", file=sys.stderr)

    except Exception as e:
        print(f"[ZSub] HATA: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
