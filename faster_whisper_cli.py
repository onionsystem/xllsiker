#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZSub Transcription Engine v2.3.1 - Subtitle Build
"""

import sys, os, argparse, json, re, time, tempfile, subprocess, threading, shutil
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

ZSUB_VERSION = "2.3.1"

SUPPORTED_LANGUAGES = {
    "auto":"Auto","tr":"Turkish","en":"English","de":"German","fr":"French",
    "es":"Spanish","it":"Italian","pt":"Portuguese","nl":"Dutch","pl":"Polish",
    "ru":"Russian","uk":"Ukrainian","ja":"Japanese","ko":"Korean","zh":"Chinese",
    "ar":"Arabic","hi":"Hindi","bn":"Bengali","sv":"Swedish","da":"Danish",
    "no":"Norwegian","fi":"Finnish","el":"Greek","cs":"Czech","ro":"Romanian",
    "hu":"Hungarian","bg":"Bulgarian","hr":"Croatian","sr":"Serbian","sk":"Slovak",
    "sl":"Slovenian","et":"Estonian","lv":"Latvian","lt":"Lithuanian","vi":"Vietnamese",
    "th":"Thai","id":"Indonesian","ms":"Malay","tl":"Filipino",
    "sw":"Swahili","af":"Afrikaans","cy":"Welsh","gl":"Galician","ca":"Catalan",
    "eu":"Basque","az":"Azerbaijani","kk":"Kazakh","uz":"Uzbek","ka":"Georgian",
    "hy":"Armenian","he":"Hebrew","ur":"Urdu","fa":"Persian","ta":"Tamil","te":"Telugu",
    "mr":"Marathi","ne":"Nepali","si":"Sinhala","km":"Khmer","my":"Burmese","lo":"Lao",
    "mn":"Mongolian","am":"Amharic","yo":"Yoruba","ha":"Hausa","mk":"Macedonian",
    "bs":"Bosnian","is":"Icelandic","mt":"Maltese","lb":"Luxembourgish",
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
    r'te\u015fekk\u00fcr ederim\.?\s*$',
    r'te\u015fekk\u00fcrler\.?\s*$',
    r'altyaz\u0131\s',
    r'subtitle',
    r'transcribed by',
    r'www\.',
    r'\.com',
    r'copyright',
    r'subtitles?\s+by',
    r'kanal[\u0131i]\s+(be\u011fen|abone)',
    r'be\u011fen(in|meyi unutmay\u0131n)',
    r'abone ol',
]

def is_hallucination(text):
    t = text.lower().strip()
    if not t:
        return True
    return any(re.search(p, t, re.IGNORECASE) for p in HALLUCINATION_PATTERNS)

SENTENCE_END = re.compile(r"[.!?]+")
NEW_SENT_GAP = 0.30

def build_srt(words, wpl):
    if not words:
        return ""
    groups = []
    current = []
    for i, w in enumerate(words):
        text = w['word'].strip()
        is_sentence_end = bool(SENTENCE_END.search(text))
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
        s = g[0]['start']
        e = g[-1]['end']
        if e <= s:
            e = s + 0.1
        subs.append({'start': s, 'end': e, 'text': t})
    for j in range(len(subs) - 1):
        gap = subs[j + 1]['start'] - subs[j]['end']
        if gap < 0.08:
            subs[j]['end'] = max(subs[j]['start'] + 0.05, subs[j + 1]['start'] - 0.08)
        elif gap > 0.15:
            # SORUN 2 FIX: Altyazi konusma bittikten sonra ekranda kalmasin.
            # end timestamp'ini bir sonraki altyazinin baslamasina max 0.1s kala kes.
            # Word-level clamp yeterli degilse burada da uyguluyoruz.
            natural_dur = subs[j]['end'] - subs[j]['start']
            # Altyazi max 1.5s + kelime sayisina gore ek sure ekle
            word_count = len(subs[j]['text'].split())
            reading_dur = word_count * 0.35  # 350ms per word minimum reading time
            max_dur = max(reading_dur, natural_dur) + 0.1
            hard_limit = subs[j + 1]['start'] - 0.1
            subs[j]['end'] = min(subs[j]['end'], subs[j]['start'] + max_dur, hard_limit)
    lines = []
    for idx, sub in enumerate(subs, 1):
        lines += [str(idx), f"{seconds_to_srt(sub['start'])} --> {seconds_to_srt(sub['end'])}", sub['text'], '']
    return '\n'.join(lines)

_model = None
_model_path = None
_dev = None
_comp = None

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
    # beam_size: accuracy vs speed
    # No BatchedInferencePipeline - use direct transcribe to prevent segment skipping
    # CPU: conservative to avoid crash
    if device != 'cuda':
        return dict(beam_size=3, batch_size=1, chunk_len=30, num_workers=1)
    if vram >= 12:
        return dict(beam_size=5, batch_size=16, chunk_len=30, num_workers=2)
    if vram >= 8:
        return dict(beam_size=5, batch_size=8, chunk_len=30, num_workers=1)
    if vram >= 4:
        return dict(beam_size=3, batch_size=4, chunk_len=30, num_workers=1)
    return dict(beam_size=2, batch_size=2, chunk_len=30, num_workers=1)

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
        path, device=_dev, compute_type=_comp,
        local_files_only=True,
        cpu_threads=min(4, os.cpu_count() or 4),
        num_workers=p['num_workers']
    )
    _model_path = path
    log(f"Model yuklendi | {_dev} vram:{vram:.1f}GB beam:{p['beam_size']}")
    return _model

def run_transcription(audio_path, model_path, language='tr', device='auto',
                      compute_type='auto', words_per_line=4, analyze_only=False):
    if not os.path.exists(model_path):
        return {'success': False, 'error': f'Model yok: {model_path}'}
    if not os.path.exists(audio_path):
        return {'success': False, 'error': f'Ses yok: {audio_path}'}

    dev, comp = detect_device(device, compute_type)
    vram = get_vram_gb() if dev == 'cuda' else 0
    p = get_params(dev, vram)
    log(f"Device:{dev} Compute:{comp} Lang:{language} beam:{p['beam_size']}")

    try:
        model = load_model(model_path, device, compute_type)
        lang_arg = language if language != 'auto' else None

        try:
            import wave
            with wave.open(audio_path, 'rb') as wf:
                audio_duration = wf.getnframes() / wf.getframerate()
        except Exception:
            audio_duration = 120.0
        log(f"Ses suresi: {audio_duration:.1f}s")

        # SORUN 1 FIX: VAD kapatildi - segment atlama sorununu cozuyor.
        # VAD bazi konusmalari yutuyor (ozellikle sessizlik sonrasi ilk kelimeler).
        # Hallucination kontrolu kelime seviyesinde yapiliyor.
        # initial_prompt: Whisper'in noktalama isaretlerini duzgun kullanmasini sagliyor (SORUN 3 FIX).
        INITIAL_PROMPT = "Konusma transkripsiyonu. Noktalama isaretlerini kullan."

        segs, info = model.transcribe(
            audio_path,
            language=lang_arg,
            word_timestamps=True,
            beam_size=p['beam_size'],
            vad_filter=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            condition_on_previous_text=True,
            temperature=0.0,
            initial_prompt=INITIAL_PROMPT,
        )

        detected = getattr(info, 'language', language)
        all_words = []
        cuts = []
        skipped = 0
        prev_end = 0.0
        SILENCE_GAP = 0.35

        for seg in segs:
            if is_hallucination(seg.text):
                skipped += 1
                log(f"  SKIP: '{seg.text.strip()[:60]}'")
                continue
            # no_speech_prob filtresi: sadece cok yuksek degerleri at, orta deger olan segmentleri tutuyoruz
            # SORUN 1 FIX: 0.6 yerine 0.85 - transkripti olan segmentleri atmiyoruz
            if hasattr(seg, 'no_speech_prob') and seg.no_speech_prob > 0.85:
                if not seg.words:
                    skipped += 1
                    continue
            if not seg.words:
                continue
            for w in seg.words:
                raw_word = w.word or ''
                if not raw_word.strip():
                    continue
                dur = max(0.0, w.end - w.start)
                prob = float(getattr(w, 'probability', 1.0) or 1.0)
                gap = max(0.0, w.start - prev_end)
                if dur <= 1.0 and (prob <= 0.90 or len(raw_word.strip()) <= 3):
                    log(f"  WORD: '{raw_word.strip()}' {w.start:.2f}-{w.end:.2f}s p:{prob:.2f}")
                if prev_end > 0 and gap >= SILENCE_GAP:
                    cuts.append({
                        'start': round(prev_end, 3),
                        'end': round(w.start, 3),
                        'reason': 'silence',
                        'dur': round(gap, 3)
                    })
                # SORUN 2 FIX: word end timestamp'larini kırp.
                # Whisper bazen word.end'i gerçek sesten cok sonraya koyuyor.
                # Makul maksimum sure: kelime uzunluguna gore hesaplanan limit.
                word_len = len(raw_word.strip())
                # Kisa kelimeler icin max 0.8s, uzun kelimeler icin max 2.0s
                max_word_dur = min(2.0, max(0.3, word_len * 0.12))
                clamped_end = w.end
                if dur > max_word_dur:
                    clamped_end = w.start + max_word_dur
                    log(f"  END-CLAMP: '{raw_word.strip()}' {w.start:.2f}-{w.end:.2f} -> {clamped_end:.2f}")
                all_words.append({'word': w.word, 'start': w.start, 'end': clamped_end})
                prev_end = clamped_end

        cuts = [c for c in cuts if c.get('dur', 0) >= 0.12]
        log(f"{len(all_words)} kelime | {skipped} atlandi | {len(cuts)} sessizlik")

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
            result['error'] = 'Kelime bulunamadi'
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
    cmd = [ffmpeg_path, '-i', audio_path, '-af',
           f'silencedetect=noise={threshold_db}dB:d={min_silence_ms/1000.0}', '-f', 'null', '-']
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
                    cuts.append({'start': round(s_start + pad, 3), 'end': round(s_end - pad, 3),
                                  'reason': 'silence', 'dur': round(dur - pad * 2, 3)})
                s_start = None
        log(f"Sessizlik: {len(cuts)} bolge ({threshold_db}dB)")
        return {'success': True, 'cuts': cuts, 'cut_count': len(cuts)}
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def find_ffmpeg():
    d = os.path.dirname(os.path.abspath(sys.argv[0]))
    for c in [os.path.join(d, 'ffmpeg.exe'), os.path.join(d, '..', 'ffmpeg.exe'),
              os.path.join(d, '..', '..', 'ffmpeg.exe')]:
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
        subprocess.run([ffmpeg_path, '-y', '-f', 'lavfi', '-t', f'{td:.3f}',
                        '-i', 'anullsrc=r=16000:cl=mono', '-acodec', 'pcm_s16le',
                        '-ar', '16000', '-ac', '1', sil], capture_output=True, timeout=60)
        cur = sil
        for i, clip in enumerate(clips):
            nxt = os.path.join(tmp, f'mix_{ts}_{i}.wav')
            dms = max(0, round(float(clip['timelineStart']) * 1000))
            fc = (f'[1:a]aformat=sample_rates=16000:channel_layouts=mono,'
                  f'adelay={dms}:all=1[d];'
                  f'[0:a][d]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]')
            r = subprocess.run([ffmpeg_path, '-y', '-i', cur,
                                 '-ss', str(clip['sourceIn']), '-t', str(clip['sourceDuration']),
                                 '-i', clip['path'], '-filter_complex', fc,
                                 '-map', '[out]', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', nxt],
                                capture_output=True, timeout=120)
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
        log(f"Ses hazirlandi: {len(clips)} klip")
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
        for h, v in [('Access-Control-Allow-Origin', '*'),
                     ('Access-Control-Allow-Methods', 'GET,POST,OPTIONS'),
                     ('Access-Control-Allow-Headers', 'Content-Type')]:
            self.send_header(h, v)
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path).path
        if p == '/health':
            self._json({'status': 'ok', 'version': ZSUB_VERSION,
                        'model_loaded': self.model_loaded,
                        'device': _dev or self.device,
                        'compute_type': _comp or self.compute_type,
                        'vram_gb': round(get_vram_gb(), 1),
                        'ffmpeg': self.ffmpeg_path is not None})
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
            self._json(run_transcription(ap, self.model_path, d.get('language', 'tr'),
                                          self.device, self.compute_type, d.get('words_per_line', 4), False))
        elif p == '/analyze':
            ap = d.get('audio_path')
            if not ap:
                self._json({'success': False, 'error': 'audio_path gerekli'}, 400)
                return
            self._json(run_transcription(ap, self.model_path, d.get('language', 'tr'),
                                          self.device, self.compute_type, 4, True))
        elif p == '/cut-by-silence':
            ap = d.get('audio_path')
            if not ap:
                self._json({'success': False, 'error': 'audio_path gerekli'}, 400)
                return
            self._json(detect_silence_simple(ap, d.get('threshold_db', -35),
                                              d.get('min_silence_ms', 500),
                                              d.get('min_cut_duration', 0.15), self.ffmpeg_path))
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
        log("Model hazir")
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
    log(f"http://127.0.0.1:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("Kapaniyor")
        srv.server_close()

def run_cli():
    pa = argparse.ArgumentParser(description='ZSub Engine v2.3 - Subtitle Build')
    pa.add_argument('-m', '--model', required=True)
    pa.add_argument('-f', '--file', required=True)
    pa.add_argument('-l', '--language', default='tr')
    pa.add_argument('--words-per-line', type=int, default=4)
    pa.add_argument('-of', '--output', required=True)
    pa.add_argument('--device', default='auto')
    pa.add_argument('--compute-type', default='auto')
    pa.add_argument('--analyze-only', action='store_true')
    a = pa.parse_args()
    r = run_transcription(a.file, a.model, a.language, a.device, a.compute_type,
                           a.words_per_line, a.analyze_only)
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
