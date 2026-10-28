# Split serving on two AMD cards: the amdgpu memory budget

This branch (`v0.2.7-pdhm-fix-v0.0`) is for confirming why Strata 0.2.7's split serving does not start on two MI50s. It holds the Strata version we tested, the tools we measured with, and the results. For running PDHyperMix itself, use [`main`](https://github.com/MarcoTool/PDHyperMix-Strata-MI50).

Upstream report: linked here once it is posted.

## Strata version

[MarcoTool/Strata `fix/gfx906-blocking-staging`](https://github.com/MarcoTool/Strata/tree/fix/gfx906-blocking-staging) at [`d39fb2cb`](https://github.com/MarcoTool/Strata/commit/d39fb2cb):

| Commit | What it is |
|---|---|
| [`5e8a4cca`](https://github.com/Niko1221/Strata/commit/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d) | Upstream Strata v0.2.7 (engine 0.1.52), unchanged |
| [`93f866a7`](https://github.com/MarcoTool/Strata/commit/93f866a7) | 0.2.7 compiles for gfx906 again (upstream PR #3986) |
| [`d39fb2cb`](https://github.com/MarcoTool/Strata/commit/d39fb2cb) | Copies into the GPU expert cache go through the HIP staging buffer on gfx906 too |

The branch without the last commit is [`bench/0.2.7-gfx906`](https://github.com/MarcoTool/Strata/tree/bench/0.2.7-gfx906), which is what "stock" means below.

## The two problems

The amdgpu driver keeps one budget for all host memory registered for AMD GPUs on the machine: RAM × 63/64 − 1.5 GiB, 91.3 GiB on our 94 GiB PC ([`amdgpu_amdkfd_gpuvm.c:126-130`](https://github.com/torvalds/linux/blob/v6.8/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gpuvm.c#L126-L130)). NVIDIA cards do not use it.

1. **Two HIP engines on one shared arena.** The driver charges the arena once per engine that registers it, so an arena that is one copy in RAM is charged twice. When the whole arena is refused, the second engine registers it slice by slice until the driver refuses a slice, which leaves nothing for the next allocation of either engine. This does not look specific to gfx906, but we have no other AMD cards to check it on.
2. **gfx906 build only.** Strata's HIP build copies experts through a small staging buffer, but the gfx906 build did not compile it in. Each expert the decode engine copied from the part of the arena it had not registered was registered by ROCr and stayed charged. Fixed by `d39fb2cb`.

## Our results

2× MI50 32 GB, 94 GiB RAM, Ubuntu 22.04, kernel 6.8, amdgpu DKMS 6.10.5, ROCm 7.14, Qwen3.8-Flash-Next IQ3_S (experts 46.8 GiB). Split serving: card 0 reads the prompt, card 1 answers, one arena in `/dev/shm`.

| Build | `STRATA_PIN_RESERVE_MIB` | Decode engine | Arena registered by the decode engine | Arena pages registered by copies |
|---|---|---|---|---|
| stock | not set | does not become ready | | |
| stock | 4096 | exits on `verify: copy stream create failed` | 36.1 GiB | 4.51 GiB (4,308 copies) |
| stock | 12288 | ready in 291 s | 26 GiB | |
| fix | not set | ready in 271 s, but the prefill engine fails on its first request | 41 GiB | 0 |
| fix | 4096 | ready in 292 s | 36.1 GiB | 0 |

Empty cells were not traced. A 16K prompt with no cache, one run each:

| | Stock, reserve 12288 | Fix, reserve 4096 |
|---|---|---|
| Prefill on card 0 | 352 tok/s | 347 tok/s |
| Hand-over, 465 MiB session (save + restore) | 1.4 s | 2.0 s |
| Decode on card 1 | 35.7 tok/s | 34.4 tok/s |

## Reproduce

Build Strata from the branch above for gfx906:

```bash
git clone -b fix/gfx906-blocking-staging https://github.com/MarcoTool/Strata.git && cd Strata
cmake -S . -B build-906 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 -DGPU_TARGETS=gfx906 -DSTRATA_NATIVE_EXPERTS=ON \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
ninja -C build-906 strata
```

For the stock build, check out `bench/0.2.7-gfx906` instead.

In the run config, turn split serving on and name the shared arena:

```json
{
  "split_serve": {"prefill_gpu": 0, "decode_gpu": 1},
  "args": ["...", "--shared-expert-arena", "/dev/shm/strata.arena"]
}
```

Start `serve/server.py` with that config, with or without `STRATA_PIN_RESERVE_MIB` in the environment, and look at the kernel log:

```bash
journalctl -k | grep "SVM mapping failed"
```

`amdgpu: SVM mapping failed, exceeds resident system memory limit` is the driver refusing a registration because the budget is used up.

## Measure what each engine is charged

[`tools/kfdtrace.c`](../tools/kfdtrace.c) is an `LD_PRELOAD` library that logs each process's KFD allocation, free and SVM calls; [`tools/kfdsum.py`](../tools/kfdsum.py) adds them up.

```bash
gcc -O2 -shared -fPIC -o libkfdtrace.so tools/kfdtrace.c -ldl -lpthread
LD_PRELOAD=$PWD/libkfdtrace.so KFDTRACE_DIR=/tmp/kfd python serve/server.py --config <config>
python3 tools/kfdsum.py /tmp/kfd/kfd-<pid>.tsv
```

Both engines inherit `LD_PRELOAD`, so each writes its own `kfd-<pid>.tsv`. The decode engine of the stock build with a 4096 reserve:

```
arena registered          36.11 GiB
arena pages by copies      4.51 GiB  (4308 calls)
other SVM ranges           2.54 GiB
GTT and userptr held       1.27 GiB
refused calls           7
```

`KFDTRACE_BT=N` also writes the call stack of the first N small arena registrations. The library does not see releases done by `munmap`, so its sums are upper bounds.

## Other AMD cards

We only have gfx906. If you run two HIP engines on one shared arena on another AMD card, for example an RX 7900 XTX or R9700, the output of `journalctl -k | grep "SVM mapping failed"` and of `kfdsum.py` for each engine would show whether problem 1 applies there. Please add it to the upstream report above.
