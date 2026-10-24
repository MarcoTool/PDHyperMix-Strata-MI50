# PDHyperMix-Strata-MI50

**English** · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md)

**Prefill / decode disaggregation on one home machine: a high-memory-bandwidth card decodes, a matrix-strong card prefills.**

[How it works](#how-it-works) · [Benchmarks](docs/BENCHMARKS.md) · [Deploy](docs/DEPLOY.md) · [Known issues](docs/LESSONS.md)
(the docs are in Traditional Chinese for now; Simplified Chinese copies are the `docs/*.zh-CN.md` files)

2× MI50 + RTX 3070, Qwen3.8-Flash-Next IQ3_S, engine: [Strata](https://github.com/Niko1221/Strata):

| | MI50 only | PDHyperMix |
|---|---|---|
| Prefill, 32K prompt | ~400 tok/s | **~1,600 tok/s** |
| Decode, 64K–258K | 30–35 tok/s | **39–40 tok/s** |
| TTFT (time to first token), 32K prompt | ~80 s | **~24 s** |
| Total time, 32K prompt + 200-token answer | 85.2 s | **27.8 s** |

This repo contains three things: the PD disaggregation approach, configs and scripts you can run as-is, and a Strata
change for gfx906 (a two-stage top-k on the decode side, currently in our
[fork branch `gfx906-ep`](https://github.com/MarcoTool/Strata/tree/gfx906-ep), to be sent upstream as a PR). It has been tested on our machine only; what to adjust for other hardware is in [DEPLOY](docs/DEPLOY.md).

### Pairing

With prefill and decode split, a decode-strong card can be paired with a prefill-strong card, each doing the part it
is good at. Decode strength is memory bandwidth and VRAM capacity; prefill strength is matrix units.

The MI50 is an example: 32 GB of HBM2 and about 1 TB/s of bandwidth (spec), cheap second-hand, and no need to replace
if you already own some. Two of them hold all 24,576 experts of Flash-Next in VRAM, which makes them strong at decode.
But gfx906 has no matrix units (no MFMA, no tensor cores): two cards prefill a 32K prompt at about 400 tok/s, and a
layer-split prefill only reaches 625 tok/s. Paired with a prefill-strong RTX 3070 (about 1,500–1,650 tok/s), the MI50s
only decode; an upgrade later only needs to replace one side. Other "decode-strong + prefill-strong" pairs should work
the same way in principle, but have not been measured.

### Engine

The reference implementation uses Strata: it can save a session and load it back, and one source tree builds both a
CUDA and a HIP backend, so a session can move between the two.

## How it works

Prefill and decode need different hardware: prefill is large batches of matrix math, decode is mostly bound by memory
bandwidth. PDHyperMix gives each part to a different card:

- **Prefill card**: an RTX 3070 runs Strata's CUDA build, prefilling at about 1,500–1,650 tok/s.
- **Decode cards**: 2× MI50 run Strata's HIP (gfx906) build, with every expert in VRAM.
- **Session handoff**: after prefill, the 3070 saves the session (KV and recurrent state) to a session file, and the
  MI50s load it and continue with decode. A 32K prompt's session is about 726 MB and moves through RAM in about 2.2 s.

![Request flow](assets/pd-flow.svg)

PD disaggregation is usually a data-center setup (multiple nodes, clusters linked by RDMA). Three things make it work
on one machine:

1. **One copy of the experts in RAM.** IQ3_S has about 47 GiB of experts; one copy per engine would exceed the 94 GB of
   RAM. Strata's `--shared-expert-arena` puts them in a shared file in `/dev/shm` (the shared expert arena), mapped by
   both engines.
2. **Both engines from one source tree.** A session carries a fingerprint of the engine settings: the version, the KV
   format and `--kv-resident` must match on both sides, or the session handoff fails.
3. **Session handoff through RAM.** Strata saves sessions with `O_DIRECT` and `fsync`. On a SATA SSD, saving and loading
   a 32K session takes 4.8 s; in `/dev/shm` it takes 2.2 s.

## Memory layout

![Memory layout](assets/memory-map.svg)

- **RTX 3070**: dense layers, MTP, the full 262K KV, and 2,980 experts; the rest are streamed from RAM during prefill.
- **MI50 #0**: dense layers, KV, MTP, and 11,771 experts.
- **MI50 #1**: 12,805 experts. Together the two hold all 24,576, so decode reads no experts from RAM.
- **RAM**: the shared expert arena (46.8 GiB), about 7 GiB for both engines and the OS, and the session file.

## Decode-side changes

Decode slows down at long context: 30 tok/s at 258K with the original settings. Two changes:

- **The whole KV in VRAM** (`--kv-resident 262144`): the KV at 262K is 3.1 GiB, and the experts it displaces move to
  MI50 #1. 4–14% faster at every context length.
- **A two-stage indexer top-k on gfx906** (fork, `STRATA_GFX906_TOPK_SPLIT=1`): each query used to get a single
  256-thread CTA, which took 12 ms of every window at 258K. The rewrite selects bit-identical results and makes this
  step 6× faster.

![Decode speed against context depth](assets/decay.svg)

## Benchmarks

Strata's prefill speed itself barely changes with context length. With this setup (3070 prefill, 2× MI50 decode):

| Prompt length | Prefill (tok/s) | Decode (tok/s) | TTFT (prefill + session handoff) |
|---|---|---|---|
| 16K | 1,533 | 43.8 | 12.2 s |
| 32K | 1,597 | 46.5 | 22.5 s |
| 64K | 1,681 | 40.0 | 42.4 s |
| 128K | 1,643 | 40.5 | 85.6 s |
| 192K | 1,589 | 39.3 | 132.6 s |
| 258K | 1,543 | 39.4 | 179.8 s |

Decode by task type (short context): chat 47.3, code 60.6, math 60.6 tok/s.

The table uses the production settings (KV fully in VRAM, two-stage top-k) with 300-token answers; TTFT leaves out the
few tokens the decode side reads after loading the session (about 0.1–0.3 s).

Against community results for the same engine and quantization, compared with a single R9700 32 GB: prefill is about
2.1–2.3× faster (32K / 128K), and decode is about half (46.5 vs 95.6 tok/s). Prefill is where the old architecture
really hurts; decode stays at 40–50 tok/s, which is enough for everyday use
([comparison](docs/BENCHMARKS.md#與其他硬體的單請求對照)). Methods and full data are in [BENCHMARKS](docs/BENCHMARKS.md).

## Hardware

| Part | Spec |
|---|---|
| Decode cards | 2× AMD Instinct MI50 32 GB (gfx906), sclk capped at 1,339 MHz, power capped at 170 W |
| Prefill card | NVIDIA RTX 3070 16 GB (modded, with FP16/BF16 unlocked, see [BENCHMARKS](docs/BENCHMARKS.md#prefill-卡)) |
| CPU / board | AMD Threadripper 1920X, ASUS PRIME X399-A, PCIe 3.0 |
| RAM | 94 GB |
| Storage | SATA SSD |
| Software | Ubuntu 22.04, kernel 6.8, ROCm (HIP) 7.14, NVIDIA driver 580 / CUDA 13 |

## Tested combinations

| Decode | Prefill | Status |
|---|---|---|
| 2× MI50 32 GB | RTX 3070 16 GB (modded, unlocked) | Measured, the numbers on this page |

Other combinations should work but have not been tested by us, such as an RTX 20 / 40 / 50 for prefill (the CUDA
image builds for them by default) or an MI60 / Radeon VII for decode. If you run another combination, please open an
issue with the output of `bench_tasks.py` and `bench_depth.py`, or send a PR that adds a row to this table.

## Requirements

- **RAM**: about 72 GB or more for IQ3_S (shared expert arena 47 GiB, both engines about 15 GB, OS about 4 GB, session
  file up to about 4 GB).
- **Prefill card**: for IQ3_S at 262K context, dense layers, KV, MTP and buffers take about 9.5 GiB; 12 GB or more is
  recommended.
- **Decode cards**: VRAM for every expert; 2× MI50 32 GB just fit IQ3_S.

The RAM and VRAM thresholds above are estimates, not measurements.

## Quick start

```bash
git clone -b gfx906-ep https://github.com/MarcoTool/Strata.git strata-src   # the Strata branch with the gfx906 top-k
cd scripts
cp pd.env.example pd.env       # Strata checkout, paths, GPU indices
./pd_build.sh                  # builds both engine images (CUDA, gfx906) from that one checkout
./pd_up.sh                     # starts the prefill engine, the decode engine and the router
# OpenAI-compatible endpoint: http://127.0.0.1:20030/v1
./pd_down.sh --free-arena      # stops everything and frees the shared expert arena
```

There are no prebuilt images: `pd_build.sh` builds them on your machine from `docker/Dockerfile.cuda` and
`docker/Dockerfile.gfx906`. The full steps are in [DEPLOY](docs/DEPLOY.md).

## Known issues

- Each engine serves one request at a time (session save/load and parallel requests exclude each other). Two requests at
  once finish only 16% sooner overall.
- Output is not guaranteed to be reproducible word for word: prefill (CUDA) and decode (HIP) round floating point
  differently.
- Any other Strata process must use the same `--shared-expert-arena`, or it allocates about 47 GB more RAM.
- Short-context chat decodes at about 47 tok/s; code and math at about 60.

The full list and fixes are in [LESSONS](docs/LESSONS.md).

## Layout

```
assets/    figures (SVG)
docker/    Dockerfile.cuda (prefill engine), Dockerfile.gfx906 (decode engine)
docs/      DEPLOY, BENCHMARKS, LESSONS
scripts/   router, engine configs, build, start and benchmark scripts
```

## License and credits

This repo is released under the MIT License (see [LICENSE](LICENSE)). The inference engine is
[Strata](https://github.com/Niko1221/Strata) (MIT, © Niko1221 and the Strata contributors). The decode-side top-k
change is currently in our fork branch and will be sent upstream as a PR.
