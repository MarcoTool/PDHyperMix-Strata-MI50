"""PD router (PDHyperMix-Strata-MI50) for Flash-Next: the prompt on pd-pre (RTX 3070), the answer on pd-dec (2x MI50), one OpenAI endpoint.

A request whose new text is long goes to pd-pre first (max_tokens 1); pd-pre's session is saved to the shared slot
directory and restored into pd-dec, then the request itself is sent to pd-dec, which reuses the whole prompt and only
writes the answer.  A short request, or a turn that continues what pd-dec already holds, goes to pd-dec directly.

Two locks make it a two-stage pipeline: one request is prefilled on pd-pre while another is answered on pd-dec.  The
restore and the answer happen under ONE hold of the decode lock, so no other request can replace pd-dec's session in
between.  Each engine takes one request at a time (Strata's slot save/restore is off with parallel requests).

Images: PD_IMAGES says where a conversation with images goes (see IMAGES_TO below).

Environment: PD_PORT (listen, 20030), PD_PRE_PORT (20061), PD_DEC_PORT (20062), PD_SLOTS (/dev/shm/pd-slots),
PD_MIN_CHARS (0), PD_IMAGES (prefill).  Standard library only.
"""
import hashlib
import http.client
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import uuid

LISTEN = ('127.0.0.1', int(os.environ.get('PD_PORT', '20030')))
PRE = ('127.0.0.1', int(os.environ.get('PD_PRE_PORT', '20061')))
DEC = ('127.0.0.1', int(os.environ.get('PD_DEC_PORT', '20062')))
SLOTS = os.environ.get('PD_SLOTS', '/dev/shm/pd-slots')
# new text past what pd-dec holds, in characters, from which the prompt goes to pd-pre; 0: every text request
MIN_CHARS = int(os.environ.get('PD_MIN_CHARS', '0'))
# where a conversation with images goes.  'prefill' (default): like text, pd-pre reads it and the session moves to
# pd-dec.  Each engine encodes the images again before it decides what it can reuse and keys a picture by a hash of its
# embeddings, so pd-dec reuses the session past an image only when both engines use the same encoder on the same card
# (DEC_ENCODER=prefill-gpu, the default; measured: a 1920x1080 screenshot in 7.2 s, 1,038 of 1,045 tokens reused).
# 'decode': straight to pd-dec, which reads the images itself - the setting for DEC_ENCODER=cpu, where the embeddings
# differ and a handoff would be read again (a small image: 14.1 s handed over, 4.6 s on pd-dec).  Text ignores this.
IMAGES_TO = os.environ.get('PD_IMAGES', 'prefill').strip().lower()
if IMAGES_TO not in ('prefill', 'decode'):
    sys.exit(f"PD_IMAGES must be 'prefill' or 'decode', not {IMAGES_TO!r}")
IMAGE_MARK = '[image:'

pre_lock = threading.Lock()
dec_lock = threading.Lock()
held = {'text': ''}   # the conversation pd-dec holds after its last answer (prompt + answer); guarded by dec_lock


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, file=sys.stderr, flush=True)


def part_text(p):
    """A content part as text; an image becomes a marker with a hash of its data, so a new image counts as new text."""
    if p.get('type') in ('image_url', 'input_image', 'image'):
        src = p.get('image_url') or p.get('source') or p.get('url') or ''
        if isinstance(src, dict):
            src = src.get('url') or src.get('data') or ''
        return IMAGE_MARK + hashlib.sha1(str(src).encode()).hexdigest()[:16] + ']'
    return p.get('text', '')


def flatten(body):
    """The request's messages as one string, built the same way every time, to compare with what pd-dec holds."""
    out = []
    for m in body.get('messages') or []:
        c = m.get('content')
        if isinstance(c, list):
            c = ''.join(part_text(p) for p in c if isinstance(p, dict))
        out.append(f"<{m.get('role')}>{c or ''}")
    return '\n'.join(out)


def common_prefix(a, b):
    k, n = 0, min(len(a), len(b))
    while k < n and a[k] == b[k]:
        k += 1
    return k


def call(host, method, path, body=None, timeout=7200):
    c = http.client.HTTPConnection(*host, timeout=timeout)
    data = json.dumps(body).encode() if body is not None else None
    c.request(method, path, body=data, headers={'Content-Type': 'application/json'} if data else {})
    return c, c.getresponse()


def call_json(host, path, body):
    c, r = call(host, 'POST', path, body)
    try:
        raw = r.read()
        try:
            return r.status, json.loads(raw or b'{}')
        except ValueError:
            return r.status, {'raw': raw[:300].decode(errors='ignore')}
    finally:
        c.close()


def prefill(body):
    """Stage 1, under pre_lock: pd-pre prefills the prompt and saves its session.  -> (file name, None) or (None, error)."""
    pre = dict(body, max_tokens=1, stream=False)
    for k in ('stream_options', 'max_completion_tokens'):
        pre.pop(k, None)
    name = f'pd-{uuid.uuid4().hex}.bin'
    with pre_lock:
        t0 = time.time()
        st, r = call_json(PRE, '/v1/chat/completions', pre)
        if st != 200:
            return None, f'prefill on pd-pre: HTTP {st} {json.dumps(r)[:300]}'
        t1 = time.time()
        st, r = call_json(PRE, '/slots/0?action=save', {'filename': name})
        if st != 200:
            return None, f'save on pd-pre: HTTP {st} {json.dumps(r)[:300]}'
    log(f'  prefill {t1 - t0:.1f}s save {time.time() - t1:.2f}s')
    return name, None


def answer_text(raw):
    """The answer's text from a JSON or SSE body (what pd-dec now holds after the prompt)."""
    try:
        if raw.lstrip().startswith(b'{'):
            return json.loads(raw)['choices'][0]['message'].get('content') or ''
        out = ''
        for line in raw.decode(errors='ignore').splitlines():
            if line.startswith('data: ') and line[6:].strip() not in ('', '[DONE]'):
                d = json.loads(line[6:])['choices'][0].get('delta') or {}
                out += d.get('content') or ''
        return out
    except (ValueError, KeyError, IndexError, TypeError):
        return ''


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):   # the client closed its keep-alive connection
            pass

    def reply(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def relay(self, method, path, body):
        """Forward to pd-dec and stream its reply back unchanged (JSON or SSE), chunked.  -> the raw reply."""
        c, r = call(DEC, method, path, body)
        try:
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in ('transfer-encoding', 'connection', 'content-length'):
                    self.send_header(k, v)
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            got = []
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                got.append(chunk)
                self.wfile.write(b'%x\r\n%s\r\n' % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b'0\r\n\r\n')
            self.wfile.flush()
            return b''.join(got)
        finally:
            c.close()

    def do_GET(self):
        c, r = call(DEC, 'GET', self.path, timeout=30)
        try:
            data = r.read()
            self.send_response(r.status)
            self.send_header('Content-Type', r.getheader('Content-Type', 'application/json'))
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        finally:
            c.close()

    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0)
        try:
            body = json.loads(self.rfile.read(n) or b'{}')
        except ValueError:
            return self.reply(400, {'error': {'message': 'invalid JSON'}})
        if not self.path.startswith('/v1/chat/completions'):
            with dec_lock:
                self.relay('POST', self.path, body)
            return
        text = flatten(body)
        # judged against what pd-dec held when the request came in; a request ahead in the queue may change that,
        # which at worst sends a prompt the long way or the short way - never a wrong answer
        keep = common_prefix(held['text'], text)
        fresh = len(text) - keep
        new_image = IMAGE_MARK in text[keep:]
        any_image = IMAGE_MARK in text
        name, err = (None, None)
        t0 = time.time()
        if IMAGES_TO == 'decode' and any_image:
            use_pd = False                     # the session would not be reused past an image (see IMAGES_TO)
        else:
            use_pd = fresh >= MIN_CHARS or new_image
        if use_pd:
            name, err = prefill(body)          # stage 1: may overlap the previous request's answer on pd-dec
            if err:
                log('PD prefill failed, pd-dec prefills the prompt itself:', err)
        with dec_lock:                         # stage 2: the restore and the answer in one hold
            t1 = time.time()
            if name:
                st, r = call_json(DEC, '/slots/0?action=restore', {'filename': name})
                try:
                    os.remove(os.path.join(SLOTS, name))
                except OSError:
                    pass
                if st != 200:
                    log('PD restore failed, pd-dec prefills the prompt itself:', st, json.dumps(r)[:300])
            t2 = time.time()
            raw = self.relay('POST', self.path, body)
            held['text'] = text + '\n<assistant>' + answer_text(raw)
        log(f'{len(text)} chars ({fresh} new{", image" if any_image else ""}) {"PD" if name else "direct"}: queue+prefill {t1 - t0:.1f}s '
            f'restore {t2 - t1:.2f}s answer {time.time() - t2:.1f}s')


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == '__main__':
    log(f'PD router on {LISTEN[0]}:{LISTEN[1]}: pre :{PRE[1]}, dec :{DEC[1]}, PD from {MIN_CHARS} new chars, images to {IMAGES_TO}')
    Server(LISTEN, Handler).serve_forever()
