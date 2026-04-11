#!/usr/bin/env python3
"""
ZSub Transcription Engine v2.1.0
CLI + HTTP Server modu — Flask bağımlılığı YOK (built-in http.server)

Kullanım:
  CLI:
    zsub-engine.exe -m model/ -f audio.wav -l tr -of output

  HTTP Server:
    zsub-engine.exe --serve -m model/
    zsub-engine.exe --serve -m model/ --port 9876
"""

import sys, os, argparse, json, re, time, tempfile, subprocess, threading, shutil
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

ZSUB_VERSION = "2.2.0"

SUPPORTED_LANGUAGES = {
    "auto":"Otomatik","tr":"Türkçe","en":"English","de":"Deutsch","fr":"Français",
    "es":"Español","it":"Italiano","pt":"Português","nl":"Nederlands","pl":"Polski",
    "ru":"Русский","uk":"Українська","ja":"日本語","ko":"한국어","zh":"中文",
    "ar":"العربية","hi":"हिन्दी","bn":"বাংলা","sv":"Svenska","da":"Dansk",
    "no":"Norsk","fi":"Suomi","el":"Ελληνικά","cs":"Čeština","ro":"Română",
    "hu":"Magyar","bg":"Български","hr":"Hrvatski","sr":"Srpski","sk":"Slovenčina",
    "sl":"Slovenščina","et":"Eesti","lv":"Latviešu","lt":"Lietuvių","vi":"Tiếng Việt",
    "th":"ไทย","id":"Bahasa Indonesia","ms":"Bahasa Melayu","tl":"Filipino",
    "sw":"Kiswahili","af":"Afrikaans","cy":"Cymraeg","gl":"Galego","ca":"Català",
    "eu":"Euskara","az":"Azərbaycan","kk":"Қазақ","uz":"O'zbek","ka":"ქართული",
    "hy":"Հայերեն","he":"עברית","ur":"اردو","fa":"فارسی","ta":"தமிழ்","te":"తెలుగు",
    "mr":"मराठी","ne":"नेपाली","si":"සිංහල","km":"ខ្មែរ","my":"မြန်မာ","lo":"ລາວ",
    "mn":"Монгол","am":"አማርኛ","yo":"Yorùbá","ha":"Hausa","mk":"Македонски",
    "bs":"Bosanski","is":"Íslenska","mt":"Malti","lb":"Lëtzebuergesch",
}

def log(msg):
    try:
        print(f"[ZSub] {msg}", file=sys.stderr)
    except (UnicodeEncodeError, OSError):
        try:
            print(f"[ZSub] {msg.encode('utf-8', errors='replace').decode('utf-8')}", file=sys.stderr)
        except Exception:
            pass

def seconds_to_srt(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = min(999, int(round((s % 1) * 1000)))
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

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

# ── Cümle sonu noktalama işaretleri (build_srt'de kullanılır) ──
SENTENCE_END = re.compile(r"[.!?]+")
# Büyük harf başlangıcı + bu kadar gap (sn) → whisper'ın atladığı cümle sınırı
NEW_SENT_GAP = 0.30

def is_hallucination(text):
    t = text.lower().strip()
    if not t:
        return True
    return any(re.search(p, t, re.IGNORECASE) for p in HALLUCINATION_PATTERNS)

STRONG_FILLERS = {
    # Tek karakter — whisper bazen "ıı" yerine tek "ı" yazıyor
    'ı', 'i', 'e', 'a',
    # Tekrarlı sesli
    'ıı', 'ııı', 'ıııı', 'ii', 'iii', 'iiii',
    'ee', 'eee', 'eeee',
    'aa', 'aaa', 'aaaa',
    # Nasal
    'mm', 'mmm', 'mmmm',
    'hm', 'hmm', 'hmmm',
    # İngilizce/evrensel
    'uh', 'uhh', 'uhhh',
    'um', 'umm', 'ummm',
    'ah', 'ahh', 'oh', 'ohh',
    'eh', 'ehh', 'em', 'ehm', 'erm',
    'ih',
}
WEAK_FILLERS = {
    'şey', 'sey',
    'yani',
    'işte', 'iste',
    'hani',
    'ya', 'ha', 'he', 'aha'
}

# Strong filler: kelime listede + kısa süre
# prob kontrolü YOK — whisper filler'lara da yüksek prob verebiliyor
# Güvenlik: tek karakter filler'larda (ı, e, a) ek kontrol var (aşağıda)
FILLER_MAX_DUR_STRONG = 0.90
FILLER_MAX_DUR_WEAK = 0.45
FILLER_LOW_PROB = 0.65
SILENCE_GAP_THRESHOLD = 0.35
MIN_CUT_DURATION = 0.12

FILLER_PROMPTS = {
    'tr': 'Şey, ıı, eee, hmm, mmm, hani, yani, işte.',
    'en': 'Um, uh, like, you know, hmm, so, erm.',
    'de': 'Ähm, äh, hmm, also, halt, ne.',
    'fr': 'Euh, hein, ben, alors, genre.',
    'es': 'Em, eh, este, bueno, pues.',
}

def _get_filler_prompt(lang):
    prompt = FILLER_PROMPTS.get(lang, FILLER_PROMPTS.get('en', ''))
    try:
        _ = prompt.encode('utf-8').decode('utf-8')
        if any(c in prompt for c in ['ı', 'ş', 'ü', 'U', 'h']):
            return prompt
    except Exception:
        pass
    if lang == 'tr':
        return 'Sey, ii, eee, hmm, mmm, hani, yani, iste.'
    return prompt

# PyInstaller binary'de Türkçe karakter bozulabilir — runtime test
def normalize_word(word):
    w = (word or '').strip().lower()
    w = re.sub(r'[^\wçğıöşü]+', '', w, flags=re.IGNORECASE)
    return w

def is_repeated_filler(w):
    if not w:
        return False
    # ı, ıı, ııı... veya i, ii, iii...
    if re.fullmatch(r'[ıi]+', w):
        return True
    # e, ee, eee...
    if re.fullmatch(r'e+', w):
        return True
    # a, aa, aaa...
    if re.fullmatch(r'a+', w):
        return len(w) >= 2
    # m, mm, mmm...
    if re.fullmatch(r'm+', w):
        return len(w) >= 2
    # hm, hmm, hmmm...
    if re.fullmatch(r'hm+', w):
        return True
    return False

def classify_filler(word, dur, prob):
    w = normalize_word(word)

    # Strong filler veya tekrarlı ses
    if w in STRONG_FILLERS or is_repeated_filler(w):
        # Tek karakter (ı, e, a, i) — gerçek kelime de olabilir
        # Ek güvenlik: prob düşük VEYA süre kısa olmalı
        if len(w) == 1:
            if dur <= 0.50 and prob <= 0.92:
                return f"filler:{w}"
            return None

        # Çoklu karakter (ıı, eee, hmm vs) — prob kontrolü yok, dur yeterli
        if dur <= FILLER_MAX_DUR_STRONG:
            return f"filler:{w or 'sound'}"

    # Weak filler: şey, yani, işte — düşük prob + kısa süre
    if w in WEAK_FILLERS:
        if dur <= FILLER_MAX_DUR_WEAK and prob <= FILLER_LOW_PROB:
            return f"filler:{w}"

    return None

def detect_device(device_arg, compute_arg):
    if device_arg == 'auto':
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            device = 'cpu'
    else:
        device = device_arg

    compute = ('int8_float16' if device == 'cuda' else 'int8') if compute_arg == 'auto' else compute_arg
    return device, compute

def get_vram_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0

def get_params(device, vram):
    if device != 'cuda':
        return dict(beam_size=3, batch_size=1, chunk_len=30, num_workers=1)
    if vram >= 12:
        return dict(beam_size=5, batch_size=16, chunk_len=30, num_workers=2)
    if vram >= 8:
        return dict(beam_size=5, batch_size=8, chunk_len=30, num_workers=1)
    return dict(beam_size=3, batch_size=4, chunk_len=25, num_workers=1)

def build_srt(words, wpl):
    """
    Kelime listesinden SRT üretir.

    Kırılım önceliği:
      1. Büyük harf başlangıcı + gap >= NEW_SENT_GAP  → whisper'ın atladığı cümle sınırı
      2. Noktalama işareti (.!?…)                     → her zaman kes, wpl'den bağımsız
      3. wpl doldu, noktalama yok                     → kullanıcı tercihi, normal kes
    """
    if not words:
        return ""

    groups = []
    current = []

    for i, w in enumerate(words):
        text = w['word'].strip()
        is_sentence_end = bool(SENTENCE_END.search(text))

        # Büyük harf + yeterli gap → whisper nokta koymadı ama cümle değişti
        is_new_sentence = False
        if current and i > 0:
            prev_text = words[i - 1]['word'].strip()
            prev_ended = bool(SENTENCE_END.search(prev_text))
            if not prev_ended and text and text[0].isupper():
                gap = w['start'] - words[i - 1]['end']
                if gap >= NEW_SENT_GAP:
                    is_new_sentence = True

        if is_new_sentence and current:
            groups.append(current)
            current = []

        current.append(w)

        if is_sentence_end:
            groups.append(current)
            current = []
        elif len(current) >= wpl:
            groups.append(current)
            current = []

    if current:
        if groups and len(current) <= 2:
            groups[-1] = groups[-1] + current
        else:
            groups.append(current)

    if len(groups) > 1 and len(groups[-1]) == 1:
        groups[-2] = groups[-2] + groups[-1]
        groups.pop()

    subs = []
    for g in groups:
        t = ' '.join(w['word'].strip() for w in g).strip()
        if not t:
            continue
        s, e = g[0]['start'], g[-1]['end']
        if e <= s:
            e = s + 0.1
        subs.append({'start': s, 'end': e, 'text': t})

    for j in range(len(subs) - 1):
        gap = subs[j + 1]['start'] - subs[j]['end']
        if gap < 0.08:
            subs[j]['end'] = max(subs[j]['start'] + 0.05, subs[j + 1]['start'] - 0.08)
        elif gap > 0.4:
            natural_dur = subs[j]['end'] - subs[j]['start']
            max_end = subs[j]['start'] + natural_dur + 0.15
            subs[j]['end'] = min(subs[j]['end'], max_end, subs[j + 1]['start'] - 0.12)

    lines = []
    for idx, sub in enumerate(subs, 1):
        lines += [
            str(idx),
            f"{seconds_to_srt(sub['start'])} --> {seconds_to_srt(sub['end'])}",
            sub['text'],
            ''
        ]
    return '\n'.join(lines)

_model = None
_model_path = None
_dev = None
_comp = None

def load_model(path, device='auto', compute='auto'):
    global _model, _model_path, _dev, _comp

    if _model and _model_path == path:
        log("Model cache")
        return _model

    _dev, _comp = detect_device(device, compute)
    vram = get_vram_gb() if _dev == 'cuda' else 0
    p = get_params(_dev, vram)

    from faster_whisper import WhisperModel
    _model = WhisperModel(
        path,
        device=_dev,
        compute_type=_comp,
        local_files_only=True,
        cpu_threads=min(4, os.cpu_count() or 4),
        num_workers=p['num_workers']
    )
    _model_path = path
    log(f"Model yüklendi | {_dev} vram:{vram:.1f}GB beam:{p['beam_size']}")
    return _model

def run_transcription(audio_path, model_path, language='tr', device='auto',
                      compute_type='auto', words_per_line=4, analyze_only=False,
                      filler_pass=False):
    if not os.path.exists(model_path):
        return {'success': False, 'error': f'Model yok: {model_path}'}
    if not os.path.exists(audio_path):
        return {'success': False, 'error': f'Ses yok: {audio_path}'}

    dev, comp = detect_device(device, compute_type)
    vram = get_vram_gb() if dev == 'cuda' else 0
    p = get_params(dev, vram)

    if filler_pass:
        log(f"=== FILLER PASS === Device:{dev} Lang:{language}")
    else:
        log(f"Device:{dev} Compute:{comp} Lang:{language}")

    try:
        model = load_model(model_path, device, compute_type)

        lang_arg = language if language != 'auto' else None
        filler_prompt = _get_filler_prompt(language)

        if filler_pass:
            # ─── FILLER PASS: VAD kapalı, sadece filler aranıyor ───
            # no_speech_threshold düşük → whisper "konuşma yok" deyip atlamıyor
            # chunk_length küçük → kısa ıı/eee seslerini daha iyi segment'e alıyor
            log(f"Filler pass prompt: {filler_prompt}")
            segs, info = model.transcribe(
                audio_path,
                language=lang_arg,
                word_timestamps=True,
                beam_size=p['beam_size'],
                vad_filter=False,
                no_speech_threshold=0.3,          # 0.6 → 0.3: ıı gibi belirsiz sesleri atlama
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                condition_on_previous_text=True,   # False → True: context bağlı daha tutarlı yazım
                temperature=0.0,
                chunk_length=20,                   # 30 → 20: kısa sesleri daha iyi yakala
                initial_prompt=filler_prompt or None
            )

            detected = getattr(info, 'language', language)
            filler_cuts = []
            filler_debug = []

            for seg in segs:
                if is_hallucination(seg.text):
                    continue
                if not seg.words:
                    continue
                for w in seg.words:
                    raw_word = w.word or ''
                    wt = normalize_word(raw_word)
                    if not wt:
                        continue
                    dur = max(0.0, w.end - w.start)
                    prob = float(getattr(w, 'probability', 1.0) or 1.0)

                    if dur <= 1.0 and (prob <= 0.90 or len(wt) <= 3):
                        log(f"  FP-WORD: '{raw_word.strip()}' norm:'{wt}' {w.start:.2f}-{w.end:.2f}s prob:{prob:.2f} dur:{dur:.2f}")

                    filler_reason = classify_filler(raw_word, dur, prob)
                    if filler_reason:
                        filler_cuts.append({
                            'start': round(w.start, 3),
                            'end': round(w.end, 3),
                            'reason': filler_reason,
                            'dur': round(dur, 3),
                            'prob': round(prob, 3),
                            'word': raw_word.strip()
                        })
                        filler_debug.append(f"  FP-FILLER: '{raw_word.strip()}' {w.start:.2f}-{w.end:.2f}s prob:{prob:.2f} dur:{dur:.2f}")

            log(f"Filler pass sonuc: {len(filler_cuts)} filler bulundu")
            for fd in filler_debug:
                log(fd)

            out_base = os.path.splitext(audio_path)[0]
            filler_path = out_base + '.fillers.json'
            with open(filler_path, 'w', encoding='utf-8') as f:
                json.dump(filler_cuts, f, ensure_ascii=False, indent=2)

            return {
                'success': True,
                'filler_path': filler_path,
                'fillers': filler_cuts,
                'filler_count': len(filler_cuts),
                'detected_language': detected,
                'error': None
            }

        # ─── NORMAL PASS: VAD açık, altyazı + sessizlik tespiti ───
        vad = dict(
            threshold=0.45,
            min_speech_duration_ms=250,
            max_speech_duration_s=float(p['chunk_len']),
            min_silence_duration_ms=500,
            speech_pad_ms=400
        )
        lang_arg = language if language != 'auto' else None

        # Filler prompt: whisper'a "bu dolgu seslerini olduğu gibi yaz" demek
        filler_prompt = _get_filler_prompt(language)
        log(f"Filler prompt ({language}): {filler_prompt}")

        if dev == 'cuda':
            try:
                from faster_whisper import BatchedInferencePipeline
                pipe = BatchedInferencePipeline(model=model)
                segs, info = pipe.transcribe(
                    audio_path,
                    language=lang_arg,
                    word_timestamps=True,
                    batch_size=p['batch_size'],
                    vad_filter=True,
                    vad_parameters=vad,
                    initial_prompt=filler_prompt or None
                )
                log("Batched mod")
            except Exception as e:
                log(f"Batched fail: {e}")
                segs, info = model.transcribe(
                    audio_path,
                    language=lang_arg,
                    word_timestamps=True,
                    beam_size=p['beam_size'],
                    vad_filter=True,
                    vad_parameters=vad,
                    no_speech_threshold=0.6,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    condition_on_previous_text=True,
                    temperature=0.0,
                    chunk_length=p['chunk_len'],
                    initial_prompt=filler_prompt or None
                )
        else:
            segs, info = model.transcribe(
                audio_path,
                language=lang_arg,
                word_timestamps=True,
                beam_size=p['beam_size'],
                vad_filter=True,
                vad_parameters=vad,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                condition_on_previous_text=True,
                temperature=0.0,
                chunk_length=p['chunk_len'],
                initial_prompt=filler_prompt or None
            )

        detected = getattr(info, 'language', language)
        all_words = []
        cuts = []
        skipped = 0
        prev_end = 0.0

        # Her filler cut'ı için debug log
        filler_debug = []

        for seg in segs:
            if is_hallucination(seg.text):
                skipped += 1
                continue
            if hasattr(seg, 'no_speech_prob') and seg.no_speech_prob > 0.6:
                skipped += 1
                continue
            if not seg.words:
                continue

            for w in seg.words:
                raw_word = w.word or ''
                wt = normalize_word(raw_word)
                if not wt:
                    continue

                dur = max(0.0, w.end - w.start)
                prob = float(getattr(w, 'probability', 1.0) or 1.0)
                gap = max(0.0, w.start - prev_end)

                # DEBUG: her kelimeyi logla (kısa kelimeler ve düşük prob önemli)
                if dur <= 1.0 and (prob <= 0.90 or len(wt) <= 3):
                    log(f"  WORD: '{raw_word.strip()}' norm:'{wt}' {w.start:.2f}-{w.end:.2f}s prob:{prob:.2f} dur:{dur:.2f}")

                if prev_end > 0 and gap >= SILENCE_GAP_THRESHOLD:
                    cuts.append({
                        'start': round(prev_end, 3),
                        'end': round(w.start, 3),
                        'reason': 'silence',
                        'dur': round(gap, 3)
                    })

                filler_reason = classify_filler(raw_word, dur, prob)
                if filler_reason:
                    cuts.append({
                        'start': round(w.start, 3),
                        'end': round(w.end, 3),
                        'reason': filler_reason,
                        'dur': round(dur, 3),
                        'prob': round(prob, 3),
                        'word': raw_word.strip()
                    })
                    filler_debug.append(f"  FILLER: '{raw_word.strip()}' {w.start:.2f}-{w.end:.2f}s prob:{prob:.2f} dur:{dur:.2f}")
                    prev_end = w.end
                    continue

                all_words.append({
                    'word': w.word,
                    'start': w.start,
                    'end': w.end
                })
                prev_end = w.end

        cuts = [c for c in cuts if c['dur'] >= MIN_CUT_DURATION]
        filler_cuts = [c for c in cuts if 'filler' in (c.get('reason') or '')]
        silence_cuts = [c for c in cuts if c.get('reason') == 'silence']
        log(f"{len(all_words)} kelime | {skipped} atlandı | {len(cuts)} kesim ({len(filler_cuts)} filler, {len(silence_cuts)} sessizlik)")
        for fd in filler_debug:
            log(fd)

        out_base = os.path.splitext(audio_path)[0]
        cuts_path = out_base + '.cuts.json'
        with open(cuts_path, 'w', encoding='utf-8') as f:
            json.dump(cuts, f, ensure_ascii=False, indent=2)

        result = {
            'success': True,
            'cuts_path': cuts_path,
            'cuts': cuts,
            'word_count': len(all_words),
            'cut_count': len(cuts),
            'segments_skipped': skipped,
            'detected_language': detected,
            'srt_path': None,
            'error': None
        }

        if analyze_only:
            return result

        if not all_words:
            result['success'] = False
            result['error'] = 'Kelime yok'
            return result

        srt_path = out_base + '.srt'
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(build_srt(all_words, words_per_line))
        result['srt_path'] = srt_path
        log(f"SRT: {srt_path}")
        return result

    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {'success': False, 'error': str(e)}

def detect_silence_simple(audio_path, threshold_db=-35, min_silence_ms=500,
                          min_cut_duration=0.15, ffmpeg_path=None):
    if not ffmpeg_path:
        ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        return {'success': False, 'error': 'FFmpeg yok'}
    if not os.path.exists(audio_path):
        return {'success': False, 'error': 'Dosya yok'}

    cmd = [
        ffmpeg_path, '-i', audio_path, '-af',
        f'silencedetect=noise={threshold_db}dB:d={min_silence_ms/1000.0}',
        '-f', 'null', '-'
    ]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        cuts = []
        s_start = None

        for line in r.stderr.split('\n'):
            m = re.search(r'silence_start:\s*([\d.]+)', line)
            if m:
                s_start = float(m.group(1))
                continue

            m = re.search(r'silence_end:\s*([\d.]+)', line)
            if m and s_start is not None:
                s_end = float(m.group(1))
                dur = s_end - s_start
                pad = 0.05
                if dur - pad * 2 >= min_cut_duration:
                    cuts.append({
                        'start': round(s_start + pad, 3),
                        'end': round(s_end - pad, 3),
                        'reason': 'silence',
                        'dur': round(dur - pad * 2, 3)
                    })
                s_start = None

        log(f"Sessizlik: {len(cuts)} bölge ({threshold_db}dB)")
        return {'success': True, 'cuts': cuts, 'cut_count': len(cuts)}
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def find_ffmpeg():
    d = os.path.dirname(os.path.abspath(sys.argv[0]))
    for c in [
        os.path.join(d, 'ffmpeg.exe'),
        os.path.join(d, '..', 'ffmpeg.exe'),
        os.path.join(d, '..', '..', 'ffmpeg.exe')
    ]:
        if os.path.exists(c):
            return os.path.abspath(c)
    return shutil.which('ffmpeg')

def prepare_audio(clips, total_duration, output_path, ffmpeg_path=None):
    if not ffmpeg_path:
        ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        return {'success': False, 'error': 'FFmpeg yok'}
    if not clips:
        return {'success': False, 'error': 'Klip yok'}

    td = float(total_duration)
    if td <= 0.1:
        for c in clips:
            e = float(c['timelineStart']) + float(c['sourceDuration'])
            if e > td:
                td = e
        td += 1.0
    if td > 36000:
        td = 60.0

    tmp = os.path.dirname(output_path) or tempfile.gettempdir()
    ts = int(time.time() * 1000)

    try:
        sil = os.path.join(tmp, f'silence_{ts}.wav')
        subprocess.run([
            ffmpeg_path, '-y', '-f', 'lavfi', '-t', f'{td:.3f}',
            '-i', 'anullsrc=r=16000:cl=mono',
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', sil
        ], capture_output=True, timeout=60)

        cur = sil
        for i, clip in enumerate(clips):
            nxt = os.path.join(tmp, f'mix_{ts}_{i}.wav')
            dms = max(0, round(float(clip['timelineStart']) * 1000))
            fc = (
                f'[1:a]aformat=sample_rates=16000:channel_layouts=mono,'
                f'adelay={dms}:all=1[d];'
                f'[0:a][d]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]'
            )
            r = subprocess.run([
                ffmpeg_path, '-y',
                '-i', cur,
                '-ss', str(clip['sourceIn']),
                '-t', str(clip['sourceDuration']),
                '-i', clip['path'],
                '-filter_complex', fc,
                '-map', '[out]',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                nxt
            ], capture_output=True, timeout=120)

            if i > 0 and os.path.exists(cur):
                try:
                    os.remove(cur)
                except Exception:
                    pass

            if r.returncode != 0:
                log(f"Klip {i} hata, skip")
                continue

            cur = nxt
            if i % 20 == 0:
                log(f"Ses: {i}/{len(clips)}")

        if os.path.exists(cur) and cur != output_path:
            shutil.move(cur, output_path)

        if os.path.exists(sil):
            try:
                os.remove(sil)
            except Exception:
                pass

        log(f"Ses birleştirildi: {len(clips)} klip")
        return {'success': True, 'output_path': output_path}
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

class ZSubHandler(BaseHTTPRequestHandler):
    model_path = None
    device = 'auto'
    compute_type = 'auto'
    model_loaded = False
    ffmpeg_path = None

    def log_message(self, fmt, *a):
        log(f"HTTP {a[0] if a else ''}")

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        l = int(self.headers.get('Content-Length', 0))
        return json.loads(self.rfile.read(l).decode('utf-8')) if l else {}

    def do_OPTIONS(self):
        self.send_response(204)
        for h, v in [
            ('Access-Control-Allow-Origin', '*'),
            ('Access-Control-Allow-Methods', 'GET,POST,OPTIONS'),
            ('Access-Control-Allow-Headers', 'Content-Type')
        ]:
            self.send_header(h, v)
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path).path
        if p == '/health':
            self._json({
                'status': 'ok',
                'version': ZSUB_VERSION,
                'model_loaded': self.model_loaded,
                'device': _dev or self.device,
                'compute_type': _comp or self.compute_type,
                'vram_gb': round(get_vram_gb(), 1),
                'ffmpeg': self.ffmpeg_path is not None
            })
        elif p == '/version':
            self._json({'version': ZSUB_VERSION})
        elif p == '/languages':
            self._json({'languages': SUPPORTED_LANGUAGES, 'count': len(SUPPORTED_LANGUAGES)})
        else:
            self._json({'error': '404'}, 404)

    def do_POST(self):
        p = urlparse(self.path).path
        try:
            d = self._body()
        except Exception as e:
            self._json({'success': False, 'error': str(e)}, 400)
            return

        if p == '/prepare-audio':
            out = d.get('output_path') or os.path.join(tempfile.gettempdir(), f'zsub_{int(time.time())}.wav')
            self._json(prepare_audio(d.get('clips', []), d.get('total_duration', 0), out, self.ffmpeg_path))
        elif p == '/transcribe':
            ap = d.get('audio_path')
            if not ap:
                self._json({'success': False, 'error': 'audio_path gerekli'}, 400)
                return
            self._json(run_transcription(
                ap, self.model_path, d.get('language', 'tr'),
                self.device, self.compute_type, d.get('words_per_line', 4), False
            ))
        elif p == '/analyze':
            ap = d.get('audio_path')
            if not ap:
                self._json({'success': False, 'error': 'audio_path gerekli'}, 400)
                return
            self._json(run_transcription(
                ap, self.model_path, d.get('language', 'tr'),
                self.device, self.compute_type, 4, True
            ))
        elif p == '/cut-by-silence':
            ap = d.get('audio_path')
            if not ap:
                self._json({'success': False, 'error': 'audio_path gerekli'}, 400)
                return
            self._json(detect_silence_simple(
                ap, d.get('threshold_db', -35), d.get('min_silence_ms', 500),
                d.get('min_cut_duration', 0.15), self.ffmpeg_path
            ))
        elif p == '/shutdown':
            log("Shutdown")
            self._json({'status': 'shutting_down'})
            threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0))).start()
        else:
            self._json({'error': '404'}, 404)

def run_server(port, model_path, device='auto', compute_type='auto'):
    log(f"Server port:{port}")
    ml = False
    try:
        load_model(model_path, device, compute_type)
        ml = True
        log("Model hazır")
    except Exception as e:
        log(f"Model hata: {e}")

    ff = find_ffmpeg()
    log(f"FFmpeg: {ff or 'YOK'}")

    ZSubHandler.model_path = model_path
    ZSubHandler.device = device
    ZSubHandler.compute_type = compute_type
    ZSubHandler.model_loaded = ml
    ZSubHandler.ffmpeg_path = ff

    srv = HTTPServer(('127.0.0.1', port), ZSubHandler)
    log(f"http://127.0.0.1:{port} — {len(SUPPORTED_LANGUAGES)} dil")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("Kapanıyor")
        srv.server_close()

def run_cli():
    pa = argparse.ArgumentParser(description='ZSub Engine')
    pa.add_argument('-m', '--model', required=True)
    pa.add_argument('-f', '--file', required=True)
    pa.add_argument('-l', '--language', default='tr')
    pa.add_argument('--words-per-line', type=int, default=4)
    pa.add_argument('-of', '--output', required=True)
    pa.add_argument('--device', default='auto')
    pa.add_argument('--compute-type', default='auto')
    pa.add_argument('--analyze-only', action='store_true')
    pa.add_argument('--filler-pass', action='store_true', help='VAD kapali filler-only tarama')
    a = pa.parse_args()

    if a.filler_pass:
        r = run_transcription(a.file, a.model, a.language, a.device, a.compute_type,
                              a.words_per_line, False, filler_pass=True)
        if r['success']:
            t = a.output + '.fillers.json'
            if r.get('filler_path') and r['filler_path'] != t:
                shutil.move(r['filler_path'], t)
            log(f"TAMAM: {r.get('filler_count', 0)} filler")
        else:
            log(f"HATA: {r.get('error')}")
            sys.exit(1)
        return

    r = run_transcription(a.file, a.model, a.language, a.device, a.compute_type, a.words_per_line, a.analyze_only)
    if r['success']:
        if r.get('srt_path'):
            t = a.output + '.srt'
            if r['srt_path'] != t:
                shutil.move(r['srt_path'], t)
        if r.get('cuts_path'):
            t = a.output + '.cuts.json'
            if r['cuts_path'] != t:
                shutil.move(r['cuts_path'], t)
        log("TAMAM")
    else:
        log(f"HATA: {r.get('error')}")
        sys.exit(1)

def main():
    # PyInstaller Windows binary'de encoding bozulmasını önle
    import io
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    if '--serve' in sys.argv:
        pa = argparse.ArgumentParser()
        pa.add_argument('--serve', action='store_true')
        pa.add_argument('--port', type=int, default=9876)
        pa.add_argument('-m', '--model', required=True)
        pa.add_argument('--device', default='auto')
        pa.add_argument('--compute-type', default='auto')
        a = pa.parse_args()
        run_server(a.port, a.model, a.device, a.compute_type)
    else:
        run_cli()

if __name__ == '__main__':
    main()
