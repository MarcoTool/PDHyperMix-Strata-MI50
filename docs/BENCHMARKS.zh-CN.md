# 测试数据与条件

所有数据均在同一台机器上、2026-10-23 到 10-24 实测。测试时机器上没有其他推理负载，每一项都注明了方法。没有实测的估算，会标明“估算”。

## 共同条件

- **模型**：Qwen3.8-Flash-Next GSQ-RCO IQ3_S（ISTA-DASLab），Strata 的 native pack，24,576 个 routed expert（48 层 × 512）
- **引擎**：Strata 0.1.43.1，fork 分支 `gfx906-ep`。prefill 端是 CUDA sm_86 build，decode 端是 HIP gfx906 build，两者同一份源码。
- **decode 端配置**：B 配置。MI50 #0 放 dense 层、KV、MTP，MI50 #1 放其余 expert（`--expert-cache-device1 auto --remote-expert-opt`）。`--spec 4 --spec-min-p 0.5 --mtp`、`--kv int8`、`--max-context 262144`、`--kv-resident 262144`。环境变量：`STRATA_GFX906_TOPK_SPLIT=1 STRATA_GR_SPLIT=1 STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1`
- **GPU 政策**：MI50 sclk 上限 1,339 MHz、功耗上限 170 W（为了散热和供电稳定，不是性能最佳值）。3070 锁频在 1,500–2,100 MHz、内存 7,001 MHz。
- **decode**：greedy（temperature 0），关闭 thinking，除非另外注明

## TTFT 与总时间：32K prompt

| | prefill | session 交接 | TTFT | decode 200 token | 总时间 |
|---|---|---|---|---|---|
| 只用 2× MI50（B 配置） | 80.2 s（400 tok/s） | — | 约 80 s | 4.8 s | 85.2 s |
| PD，session 文件放 SATA SSD | 21.2 s（1,525 tok/s） | 存 2.31 s + 读 2.50 s | 约 26 s | 4.3 s | 30.4 s |
| **PD，session 文件放 `/dev/shm`** | 21.2 s | 存 1.23 s + 读 0.94 s | **约 24 s** | 4.4 s | **27.8 s** |

TTFT 是 prefill 加上 session 交接，再加上 decode 端读入 session 后补读的 7 个 token（约 0.3 s）。

32K 的 session 文件是 725,748,212 bytes。MI50 读进来之后，32,057 个 token 里重用了 32,050 个，自己只补读 7 个。

## prefill 速度随深度（3070，CUDA build）

| prompt | 4K | 16K | 24K | 32K | 40K | 48K | 64K | 96K | 128K | 192K | 258K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tok/s | 1,053 | 1,520 | 1,558 | 1,600 | 1,603 | 1,600 | 1,598 | 1,581 | 1,549 | 1,496 | 1,439 |

（第一轮，`--kv-resident 32768`。KV 全放 VRAM 之后重量：32K 1,645、128K 1,651、258K 1,548。）

对照：只用两张 MI50 prefill，A 配置（分层）32K 是 622 tok/s，B 配置是 396 tok/s。

## decode 随深度（MI50，300 token 回答，PD）

| | 短 | 32K | 64K | 128K | 192K | 258K |
|---|---|---|---|---|---|---|
| 原始配置（`--kv-resident 32768`） | 39.0 | 42.5 | — | 34.7 | — | 30.1 |
| KV 全放 VRAM | 40.7 | 44.7 | — | 39.6 | — | 34.4 |
| **KV 全放 VRAM + 两段式 top-k** | 39.2 | 46.5 | 40.0 | 40.5 | 39.3 | 39.4 |

题目是用 tokenizer 从一份 28.2 万 token 的语料（文件加源码，没有重复内容）精确切出各深度，再加上“请详细说明上文…至少 400 字”。表中的“短”是一题介绍台湾夜市的题目，MTP 在这题的命中率偏低（110/198），所以短 prompt 的数字比下一节的一般对话低。

## decode 按任务类型（MI50，PD 的 decode 引擎直接请求）

每类 5 题，每题最多 400 token，greedy：

| 类型 | 每个 window 产出的 token | decode tok/s |
|---|---|---|
| 一般对话 | 2.13 | 47.3 |
| 程序 | 3.26 | **60.6** |
| 数学 | 3.30 | **60.6** |

每个 window 产出 = decode token 数 ÷（decode token 数 − 草稿接受数）。程序和数学的内容可预测性高，MTP 猜中的比例也高。

程序和数学题就是 `scripts/bench_tasks.py` 里的题目。一般对话的 47.3 是用我们内部的另一组 5 题测的；脚本里的一般对话题目是另外写的，用 Docker 镜像版的引擎（`docker/`）跑整个脚本：一般对话 43.4（每个 window 1.93）、编程 61.7、数学 61.1 tok/s。

## 与其他硬件的单请求对照

只列同一个引擎（Strata）、同一个量化（Flash-Next IQ3_S）、单一请求的实测。

| 硬件 | 32K prefill（tok/s） | decode，短～32K（tok/s） | decode，128K（tok/s） | 来源 |
|---|---|---|---|---|
| RTX 5090 | 4,168 | 142.2 | 127.2 | [社区，2026-10-02](https://github.com/Niko1221/Strata/tree/main/bench/results/2026-10-02-community-rtx-5090-iq3s) |
| 2× TITAN RTX | 1,642 | 80.2 | 70.6 | [社区，2026-10-08](https://github.com/Niko1221/Strata/tree/main/bench/results/2026-10-08-community-2x-titan-rtx-0.1.40.3) |
| R9700 32 GB | 750 | 95.6 | 87.6 | [社区，2026-10-08](https://github.com/Niko1221/Strata/tree/main/bench/results/2026-10-08-community-r9700-linux) |
| **PDHyperMix（2× MI50 + 3070）** | **1,597–1,645** | **46.5（32K）；一般对话 47.3／程序 60.6** | **40.5** | 本页 |
| Strix Halo（台式机） | 约 1,250（8K） | 59.7（8K） | 42.8 | [Strata 文档 STRIX_HALO.md](https://github.com/Niko1221/Strata/blob/main/docs/STRIX_HALO.md) |
| RX 9070 XT 16 GB | 601 | 45–46 | — | [社区，2026-10-02](https://github.com/Niko1221/Strata/tree/main/bench/results/2026-10-02-community-rx9070xt-windows)（引擎 0.1.35） |
| 只用 2× MI50 | 396 | 约 48 | 34.7 | 本页 |

单请求时，PDHyperMix 的 prefill 接近 2× TITAN RTX，decode 接近 Strix Halo、RX 9070 XT 这一级。

这是粗略对照：

- 各份报告的引擎版本（0.1.35–0.1.43）、生成长度（256 或 300 token）和题目都不同。
- decode 受 MTP 命中率影响很大，而命中率取决于题目；我们自己的数字就落在 39–60 tok/s 之间。
- Strix Halo 那一列取自 Strata 文档的基准值，不是社区报告；8K prefill 不等于 32K prefill。
- 量化不同的报告（Strix Halo 笔记本的 UD-IQ4_XS、RX 9070 的 IQ3_XXS、V100 的 Q2_0 等、单张 MI50 的 IQ2_XS）没有放进来。

## 两段式 top-k（gfx906 fork）

`qsa_select_bench`，3 个 query，容量 262,144，单次调用时间：

| 上下文 | 原本（ref） | wide kernel | **两段式** |
|---|---|---|---|
| 258K | 0.743 ms | 0.324 ms | **0.119 ms** |
| 131K | 0.533 ms | 0.211 ms | **0.086 ms** |
| 64K | 0.309 ms | 0.114 ms | **0.082 ms** |
| 40K | 0.211 ms | 0.092 ms | **0.076 ms** |

挑出的 id 在所有长度都跟 ref 完全一样。放进引擎实测：258K 的 `scores+topk` 从每个 window 12.24 ms 降到 4.28 ms，decode 从 33.7 提升到 39.4 tok/s，固定 prompt 的 greedy 输出跟 ref build 逐字一致。

## 交接与 RAM

- **session 文件大小**：32K 约 0.73 GB、64K 1.24 GB、128K 2.24 GB、258K 4.18 GB
- **保存加加载**（`/dev/shm`，两轮的范围）：32K 2.1–2.3 s、128K 5.3–5.5 s、258K 10.0–11.6 s
- **SATA SSD 冷读**：533 MB/s。session 文件用 `O_DIRECT` 写入，page cache 里是 0 页。
- **RAM**（两个引擎在线时）：used 7 GB、shared 50 GB（共享 expert 区）、available 约 34 GB

## 两个请求同时进来

两个不同的 32K prompt，各 decode 300 token，经路由同时或依序送出：

| | 合计 | A：TTFT / 完成 | B：TTFT / 完成 |
|---|---|---|---|
| 依序 | 62.2 s | 21.2 / 28.7 s | 26.0 / 33.5 s |
| 同时 | 52.3 s | 21.7 / 41.7 s | 44.7 / 52.1 s |

同时送时，A 的 decode 从 7.4 s 拖到 20.3 s，这段时间 3070 正在读 B 的 prompt。估算原因是 host 内存或 PCIe 的争用，还没验证。

## prefill 卡

RTX 3070 是魔改的 16 GB 版。它也用 [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) 解锁了 FP16/BF16 的指令发射率：cuBLAS 8192³ GEMM 实测 FP16 80.4、BF16 81.8、TF32 44.3 TFLOPS。

- 这张卡解锁前没有测过，所以**解锁对 prefill 的贡献没有拆开测**。
- 依源码估算：dense 层的 GEMM 和 QSA attention 走 FP16/BF16，会受益；MoE expert 走 INT8，不受影响。
- 不解锁的 3070，或其他 prefill 卡，PD 一样能运作，只是 prefill 的数字会不同。

## 试过、没有采用的方向

| 方向 | 结果 |
|---|---|
| 不改输出的融合开关（`LFUSE`、`HEAD_MIX_MULTI`、`VERIFY_QDEDUP`、`SH_STREAM`、`FUSE_CONVAB`） | window 时间差在 0.5 ms 以内 |
| `--spec` 3／5／6、`--spec-min-p` 0.3／0.7 | 换算后 46–50 tok/s，差在 3% 以内 |
| 近似路由（`STRATA_ROUTE_RESIDENT`、`STRATA_ROUTE_PRIOR`） | 更慢（43.9–46.2 对 48.7），输出相似度只有 0.19–0.32 |
| dataflow 侧流（`STRATA_DF_BRANCH`） | 44.65，变慢 |
| DFlash2 草稿模型（llama.cpp，抓取方式已修正） | 每个 window 产出：猜 3 个时 2.18／3.23／3.26（一般／程序／数学），MTP 是 2.53／3.52／3.52；猜 7 个时程序题 4.99，但验证 8 个 token 的成本估算会抵消掉 |
| 在 MI50 上做分层 prefill | 32K 从 403 提升到 625 tok/s，但远低于 3070 的 1,600 |
