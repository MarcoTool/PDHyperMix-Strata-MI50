# RTX 3070 + MI50 on Strata 0.2.7's built-in split serving

This page is for running the PDHyperMix setup (prefill on an RTX 3070, decode on MI50s) through Strata 0.2.7's own split serving, with no router outside the server. It is not optimized; it only gets the setup to run on 0.2.7.

## Strata version

[MarcoTool/Strata `split-serve-per-role`](https://github.com/MarcoTool/Strata/tree/split-serve-per-role):

| Commit | What it is |
|---|---|
| [`5e8a4cca`](https://github.com/Niko1221/Strata/commit/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d) | Upstream Strata v0.2.7 (engine 0.1.52), unchanged |
| [`93f866a7`](https://github.com/MarcoTool/Strata/commit/93f866a7) | 0.2.7 compiles for gfx906 again (upstream PR #3986) |
| [`f5cc0ac0`](https://github.com/MarcoTool/Strata/commit/f5cc0ac0) | `split_serve` takes `prefill_engine` / `decode_engine`: an executable, backend, library folders and environment per role |
| [`e27074a7`](https://github.com/MarcoTool/Strata/commit/e27074a7) | Copies into the GPU expert cache go through the HIP staging buffer on gfx906 too ([KFD_BUDGET.md](KFD_BUDGET.md)) |

Built-in split serving in 0.2.7 starts both engines from one config with only the card changed, so both run the same executable on the same backend. `f5cc0ac0` lets each role name its own, which is what upstream's [SPLIT_SERVE.md](https://github.com/Niko1221/Strata/blob/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d/docs/SPLIT_SERVE.md) lists under "NVIDIA with AMD".

> [!IMPORTANT]
> The hand-over from the CUDA engine to the gfx906 engine is accepted only because the gfx906 build writes `cuda` as the backend in its session files. The session's config fingerprint takes the backend from `STRATA_USE_HIP` ([`generate.cpp:10370-10374`](https://github.com/Niko1221/Strata/blob/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d/src/program/generate.cpp#L10370-L10374)), which the gfx906 build does not define. Upstream does not allow a cross-backend restore; on a wave32 HIP card the same hand-over would be refused (read from the code, not tested). PDHyperMix's `main` relies on the same behaviour.

## Build

Two builds of the same branch: CUDA for the RTX 3070, gfx906 for the MI50s.

```bash
git clone -b split-serve-per-role https://github.com/MarcoTool/Strata.git && cd Strata
cmake -S . -B build-cuda86 -G Ninja -DSTRATA_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \n  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
ninja -C build-cuda86 strata
cmake -S . -B build-906 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 -DGPU_TARGETS=gfx906 -DSTRATA_NATIVE_EXPERTS=ON \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
ninja -C build-906 strata
```

## Run config

The base config is the decode engine's (gfx906); `prefill_engine` switches the prefill role to the CUDA build. The two cards may have the same number because each backend numbers its own.

```json
{
  "exe": "build-906/strata",
  "backend": "hip",
  "lib_dirs": ["/opt/rocm/lib"],
  "args": ["...", "--shared-expert-arena", "/dev/shm/strata.arena"],
  "split_serve": {
    "prefill_gpu": 0,
    "decode_gpu": 0,
    "prefill_args": ["--vram-reserve-mib", "2000", "--prompt-cache", "48", "--prompt-cache-every", "8192"],
    "decode_args": ["--expert-cache-device1", "auto", "--remote-expert-opt"],
    "prefill_engine": {"exe": "build-cuda86/strata", "backend": "cuda", "lib_dirs": ["/usr/local/cuda/lib64"]},
    "decode_engine": {"env": {"HIP_VISIBLE_DEVICES": "0,1", "STRATA_GR_SPLIT": "1",
                              "STRATA_GFX906_ATTN_QUERY_SWIZZLE": "1", "STRATA_GFX906_ATTN_REDUCE12": "1"}}
  }
}
```

Start `serve/server.py --config <it>` in an environment that sees both cards (in Docker: a ROCm image with `--gpus`, `/dev/kfd` and `/dev/dri`, and the CUDA runtime libraries in `lib_dirs`) and with `HSA_OVERRIDE_GFX_VERSION=9.0.6`. The startup line says `split serving: the prompt is read on card 0, the answer is made on card 0`, and each hand-over logs a `split:` line.

## Measured

2× MI50 32 GB, RTX 3070 16 GB (VRAM modified, FP16/BF16 unlocked), 94 GiB RAM, Ubuntu 22.04, ROCm 7.14. Qwen3.8-Flash-Next IQ3_S, 256K context, `--kv int8`, `--kv-resident 262144`, MTP on. One request per length, no cache, 300 answer tokens. Every request was handed over; none fell back to the decode engine.

**Whole request, s**

| Prompt | Prefill on RTX 3070, decode on two MI50s | Prefill and decode on RTX 3070 |
|---|---|---|
| 16K | 23.4 | 23.5 |
| 32K | 31.4 | 29.2 |
| 64K | 52.3 | 47.4 |
| 128K | 95.8 | 88.2 |
| 192K | 144.2 | 130.2 |
| 258K | 189.7 | 175.1 |

**Decode, tok/s**

| Prompt | Prefill on RTX 3070, decode on two MI50s | Prefill and decode on RTX 3070 |
|---|---|---|
| 16K | 43.9 | 35.4 |
| 32K | 47.5 | 38.0 |
| 64K | 38.0 | 36.4 |
| 128K | 44.1 | 34.9 |
| 192K | 39.2 | 33.6 |
| 258K | 38.5 | 33.6 |

**Hand-over, s**

| Prompt | Session, MiB | Save | Restore | Total |
|---|---|---|---|---|
| 16K | 465 | 1.15 | 0.74 | 1.89 |
| 32K | 703 | 1.47 | 1.16 | 2.63 |
| 64K | 1,181 | 2.03 | 2.01 | 4.04 |
| 128K | 2,135 | 3.45 | 3.59 | 7.04 |
| 192K | 3,090 | 5.07 | 5.73 | 10.80 |
| 258K | 3,985 | 5.18 | 7.25 | 12.43 |

These runs used the gfx906 build without `e27074a7`. With it, 16K and 32K were also handed over: whole request 23.7 s and 32.2 s, decode 41.4 and 41.3 tok/s.
