"""Write the two engine configs (pd-pre.json, pd-dec.json) from pd.env, so both engines get identical settings.

The session snapshot carries a fingerprint of the engine settings; a snapshot saved by one engine is refused by the
other if any of them differ.  Only what MUST differ is set per engine: the libraries, the decode side's second-GPU expert
tier.  Both images keep the engine at /opt/strata/engine/strata (docker/Dockerfile.*).
"""
import json
import os
import sys


def env_file(path):
    vals = {}
    for line in open(path, encoding='utf-8'):
        line = line.split('#', 1)[0].strip()
        if '=' in line:
            k, v = line.split('=', 1)
            vals[k.strip()] = v.strip().strip('"')
    return vals


E = env_file(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'pd.env'))
common = ['--pack', E['PACK'], '--native', E['SHARD1'], '--ple-gguf', E['SHARD2'],
          '--expert-profile', E['PROFILE'], '--expert-cache', 'auto', '--prefill', 'auto',
          '--spec', E['SPEC'], '--spec-min-p', E['SPEC_MIN_P'], '--mtp', E['MTP'],
          '--max-context', E['MAX_CONTEXT'], '--kv', 'int8', '--kv-resident', E['KV_RESIDENT'],
          '--shared-expert-arena', E['ARENA']]
base = {'cwd': '/opt/strata', 'exe': '/opt/strata/engine/strata', 'tokenizer': E['PACK'] + '/tokenizer', 'model_name': 'qwen3.8-flash-next-iq3_s', 'port': 8080,
        'sampling': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0}}
pre = dict(base, args=common + ['--vram-reserve-mib', '700'],
           log=f"{E['WORK']}/pd-pre.log", lib_dirs=['/usr/local/cuda/bin', '/usr/local/cuda/lib64'], gpu=0, gpus_asked=True)
dec = dict(base, args=common + ['--expert-cache-device1', 'auto', '--remote-expert-opt'],
           log=f"{E['WORK']}/pd-dec.log", lib_dirs=['/opt/rocm/lib'], backend='hip')
# images: both engines get --vision (it is part of the session fingerprint, so the handoff needs it on both sides) and an
# encoder.  The engine keys a cached picture by a hash of its embeddings, so a session handed over past an image is
# reused only if both engines produce bit-identical embeddings.  DEC_ENCODER=prefill-gpu (default) runs the decode
# engine's encoder on the prefill card with the same CUDA binary as the prefill engine (pd_up.sh copies it out of
# IMG_PRE), which gives the same bits; DEC_ENCODER=cpu runs it on the CPU, which does not, so use PD_IMAGES=decode then.
# Without MMPROJ the engines stay text only.
SHARED_ENCODER = E.get('MMPROJ') and E.get('DEC_ENCODER', 'prefill-gpu') == 'prefill-gpu'
if E.get('MMPROJ'):
    enc = {'exe': '/opt/strata/engine/strata-vision', 'mmproj': E['MMPROJ'], 'model': E['SHARD1'], 'max_tokens': 1024}
    pre['vision'] = dict(enc, gpu=True)
    if SHARED_ENCODER:
        dec['vision'] = dict(enc, exe=f"{E['WORK']}/vision-cuda/strata-vision.sh", gpu=True)
        # the second encoder takes about 1.2 GiB of the prefill card: leave room for it beside the prefill engine
        i = pre['args'].index('--vram-reserve-mib')
        pre['args'][i + 1] = str(int(pre['args'][i + 1]) + 1300)
    else:
        dec['vision'] = dict(enc, threads=6)
    pre['args'] = pre['args'] + ['--vision']
    dec['args'] = dec['args'] + ['--vision']
# the decode engine's side session for side requests (the router's PD_SCRATCH): DEC_SCRATCH_CONTEXT cells beside the
# main session, and 500 MiB more of the first MI50 kept out of the expert cache for its prompt path and verifier
SCRATCH_CONTEXT = int(E.get('DEC_SCRATCH_CONTEXT') or 0)
if SCRATCH_CONTEXT > 0:
    dec['args'] = dec['args'] + ['--scratch-context', str(SCRATCH_CONTEXT), '--vram-reserve-mib', '1200']
os.makedirs(E['WORK'], exist_ok=True)
json.dump(pre, open(f"{E['WORK']}/pd-pre.json", 'w'), indent=1)
json.dump(dec, open(f"{E['WORK']}/pd-dec.json", 'w'), indent=1)
print(f"wrote {E['WORK']}/pd-pre.json and pd-dec.json (kv-resident {E['KV_RESIDENT']}, max-context {E['MAX_CONTEXT']}, "
      f"images {('on, decode encoder ' + ('on the prefill GPU' if SHARED_ENCODER else 'on the CPU')) if E.get('MMPROJ') else 'off'}, "
      f"side session {SCRATCH_CONTEXT if SCRATCH_CONTEXT > 0 else 'off'})")
