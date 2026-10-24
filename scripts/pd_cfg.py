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
os.makedirs(E['WORK'], exist_ok=True)
json.dump(pre, open(f"{E['WORK']}/pd-pre.json", 'w'), indent=1)
json.dump(dec, open(f"{E['WORK']}/pd-dec.json", 'w'), indent=1)
print(f"wrote {E['WORK']}/pd-pre.json and pd-dec.json (kv-resident {E['KV_RESIDENT']}, max-context {E['MAX_CONTEXT']})")
