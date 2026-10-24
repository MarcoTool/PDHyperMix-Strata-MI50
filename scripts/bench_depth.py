"""Prefill and decode against context depth, through the PD split (not the router: each step is timed on its own).

For each depth: the prefill engine prefills the prompt (max_tokens 1), saves its session, the decode engine restores it
and decodes a 300-token answer.  Prompts are cut from one long corpus at exact token counts (no repeated text, which
would let the drafter copy and inflate the speed).

    python bench_depth.py corpus.txt 4096 32768 131072 258048

Needs Strata's tokenizer (tools/strata_tokenizer.py) to cut the corpus: set STRATA_SRC and the pack's tokenizer dir.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

PRE = int(os.environ.get('PD_PRE_PORT', '20061'))
DEC = int(os.environ.get('PD_DEC_PORT', '20062'))
TOKENIZER = os.environ.get('TOKENIZER_DIR', '/opt/strata-data/packs/iq3_s/tokenizer/')
sys.path.insert(0, os.path.join(os.environ.get('STRATA_SRC', '/opt/strata-src'), 'tools'))
Q = '\n\n請詳細說明上文的內容、結構與重點，至少寫 400 字。'


def tokenizer():
    import strata_tokenizer as ST
    vocab = json.load(open(TOKENIZER + 'vocab.json', encoding='utf-8'))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    merges = open(TOKENIZER + 'merges.txt', encoding='utf-8').read().split('\n')
    return ST.Tokenizer(toks, merges, json.load(open(TOKENIZER + 'token_type.json')))


def post(port, path, body):
    t0 = time.time()
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=7200))
    except urllib.error.HTTPError as e:
        r = {'http_error': e.code, 'body': e.read().decode(errors='ignore')[:300]}
    return r, time.time() - t0


def chat(port, msgs, n):
    return post(port, '/v1/chat/completions', {'model': 'x', 'messages': msgs, 'max_tokens': n, 'temperature': 0,
                                                'chat_template_kwargs': {'enable_thinking': False}})


tk = tokenizer()
ids = tk.encode(open(sys.argv[1], encoding='utf-8', errors='ignore').read())
print('depth\tprefill tok/s\tprefill s\tsave s\trestore s\tdecode tok/s\taccepted/drafted', flush=True)
for depth in [int(x) for x in sys.argv[2:]]:
    if depth > len(ids):
        print(depth, 'corpus too short', len(ids))
        continue
    msgs = [{'role': 'user', 'content': f'[{uuid.uuid4()}]\n' + tk.decode(ids[:depth]) + Q}]
    r, t_pre = chat(PRE, msgs, 1)
    tp = r.get('timings', {})
    r1, t_save = post(PRE, '/slots/0?action=save', {'filename': 'depth.bin'})
    r2, t_rest = post(DEC, '/slots/0?action=restore', {'filename': 'depth.bin'})
    if 'http_error' in r1 or 'http_error' in r2:
        print(depth, 'handoff failed', r1, r2)
        continue
    r, _ = chat(DEC, msgs, 300)
    td = r.get('timings', {})
    print(f"{depth}\t{tp.get('prompt_per_second') or 0:.0f}\t{t_pre:.1f}\t{t_save:.2f}\t{t_rest:.2f}\t"
          f"{td.get('predicted_per_second') or 0:.1f}\t{td.get('draft_n_accepted')}/{td.get('draft_n')}", flush=True)
