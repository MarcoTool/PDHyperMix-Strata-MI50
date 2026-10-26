"""PD router (PDHyperMix-Strata-MI50) for Flash-Next: the prompt on pd-pre (RTX 3070), the answer on pd-dec (2x MI50), one OpenAI endpoint.

A request whose new text is long goes to pd-pre first (max_tokens 1); pd-pre's session is saved to the shared slot
directory and restored into pd-dec, then the request itself is sent to pd-dec, which reuses the whole prompt and only
writes the answer.  A short request, or a turn that continues what pd-dec already holds, goes to pd-dec directly.

Two locks make it a two-stage pipeline: one request is prefilled on pd-pre while another is answered on pd-dec.  The
restore and the answer happen under ONE hold of the decode lock, so no other request can replace pd-dec's session in
between.  Each engine takes one request at a time (Strata's slot save/restore is off with parallel requests).

Images: PD_IMAGES says where a conversation with images goes (see IMAGES_TO below).

Incremental handover (PD_DELTA, on by default): when a prompt continues the one pd-dec last restored, pd-pre writes
only the K/V past that point plus the running state, about 0.25 GB, instead of the whole session (2.1 GB at 128K
tokens).  pd-dec checks that it holds the file's first tokens before applying it; when it does not, pd-pre writes the
whole session and that is restored instead.

Side requests (PD_SCRATCH, off by default; needs pd-dec started with --scratch-context): a short request that shares
little with the main conversation - a title, a helper's call - runs on pd-dec's side session and never touches pd-pre
or the main conversation's cells.

Environment: PD_PORT (listen, 20030), PD_PRE_PORT (20061), PD_DEC_PORT (20062), PD_SLOTS (/dev/shm/pd-slots),
PD_MIN_CHARS (0), PD_IMAGES (prefill), PD_DELTA (1), PD_SCRATCH (0), PD_SCRATCH_MAIN (30000), PD_SCRATCH_MAX (60000),
PD_SCRATCH_CTX (32768).  Standard library only.
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
# what pd-dec holds from its last restore: the prompt text and its token count (0: nothing known); guarded by dec_lock
restored = {'text': '', 'tokens': 0}
DELTA = os.environ.get('PD_DELTA', '1') != '0'
# side requests go to pd-dec's side session only while a main conversation of PD_SCRATCH_MAIN characters or more is
# held, when they share less than half of their text with it, and when the prompt, counted as one token per character
# (CJK text; English needs fewer), plus the output asked for fits in 90% of PD_SCRATCH_CTX (pd-dec's --scratch-context)
SCRATCH = os.environ.get('PD_SCRATCH', '0') != '0'
SCRATCH_MAIN = int(os.environ.get('PD_SCRATCH_MAIN', '30000'))
SCRATCH_MAX = int(os.environ.get('PD_SCRATCH_MAX', '60000'))
SCRATCH_CTX = int(os.environ.get('PD_SCRATCH_CTX', '32768'))


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


def prefill(body, base=0):
    """Stage 1, under pre_lock: pd-pre prefills the prompt and saves its session - only past `base` tokens when pd-dec
    holds them (an incremental file).  -> (file name, None) or (None, error)."""
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
        st, r = call_json(PRE, '/slots/0?action=save', dict({'filename': name}, **({'base': base} if base else {})))
        if st != 200:
            return None, f'save on pd-pre: HTTP {st} {json.dumps(r)[:300]}'
    log(f'  prefill {t1 - t0:.1f}s save {time.time() - t1:.2f}s ({r.get("n_written", 0) / 2**20:.0f} MiB'
        f'{f", from token {base}" if base else ""})')
    return name, None


def restore(name):
    """pd-dec loads the file, which is then removed.  -> (HTTP status, body)."""
    st, r = call_json(DEC, '/slots/0?action=restore', {'filename': name})
    try:
        os.remove(os.path.join(SLOTS, name))
    except OSError:
        pass
    return st, r


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
        want = body.get('max_completion_tokens') or body.get('max_tokens') or 4096
        if (SCRATCH and len(restored['text']) >= SCRATCH_MAIN and len(text) <= SCRATCH_MAX and
                len(text) + int(want) <= SCRATCH_CTX * 0.9 and common_prefix(restored['text'], text) * 2 < len(text)):
            t0 = time.time()
            with dec_lock:                     # pd-dec's side session; pd-pre and the main conversation stay as they are
                t1 = time.time()
                self.relay('POST', self.path, dict(body, strata_lane='scratch'))
            log(f'{len(text)} chars side request (scratch): queue {t1 - t0:.1f}s answer {time.time() - t1:.1f}s')
            return
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
        # incremental when this prompt continues the one pd-dec last restored.  64 tokens of margin: where the earlier
        # prompt ends, the new text may tokenize the seam differently; pd-pre rounds the base down to a K/V page
        base = (max(0, restored['tokens'] - 64)
                if DELTA and restored['tokens'] and text.startswith(restored['text']) else 0)
        if use_pd:
            name, err = prefill(body, base)    # stage 1: may overlap the previous request's answer on pd-dec
            if err:
                log('PD prefill failed, pd-dec prefills the prompt itself:', err)
        with dec_lock:                         # stage 2: the restore and the answer in one hold
            t1 = time.time()
            if name:
                st, r = restore(name)
                if st != 200 and base:
                    # pd-dec does not hold the start the incremental file needs: the whole session instead
                    log('PD incremental restore refused, the whole session follows:', st, json.dumps(r)[:200])
                    name, err = prefill(body)
                    st, r = restore(name) if name else (0, {'error': err})
                if st == 200:
                    restored.update(text=text, tokens=int(r.get('n_restored') or 0))
                else:
                    restored.update(text='', tokens=0)
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
