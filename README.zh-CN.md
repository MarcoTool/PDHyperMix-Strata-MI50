# PDHyperMix-Strata-MI50

[English](README.md) · [繁體中文](README.zh-TW.md) · **简体中文**

**在一台家用机上做 prefill / decode 分离：高显存带宽的卡负责 decode，矩阵算力强的卡负责 prefill。**

[工作原理](#工作原理) · [实测数据](docs/BENCHMARKS.zh-CN.md) · [部署](docs/DEPLOY.zh-CN.md) · [已知问题](docs/LESSONS.zh-CN.md)

2× MI50 + RTX 3070，Qwen3.8-Flash-Next IQ3_S，引擎是 [Strata](https://github.com/Niko1221/Strata)：

| | 只用 MI50 | PDHyperMix |
|---|---|---|
| prefill，32K prompt | 约 400 tok/s | **约 1,600 tok/s** |
| decode，64K–258K | 30–35 tok/s | **39–40 tok/s** |
| TTFT（time to first token），32K prompt | 约 80 s | **约 24 s** |
| 32K prompt + 200 token 回答的总耗时 | 85.2 s | **27.8 s** |

这个仓库包含三样东西：PD 分离的做法、可以直接照着跑的配置和脚本，以及一项 Strata 的 gfx906 改动
（decode 端的两段式 top-k，目前在我们的 [fork 分支 `gfx906-ep`](https://github.com/MarcoTool/Strata/tree/gfx906-ep) 里，之后会向上游提 PR）。
目前只在我们这台机器上测过，换硬件时需要调整的地方写在 [DEPLOY](docs/DEPLOY.zh-CN.md)。

### 搭配方式

prefill 和 decode 拆开之后，可以把一张 decode 强的卡和一张 prefill 强的卡搭在一起，各做擅长的那一段。
decode 强，看的是显存带宽和显存容量；prefill 强，看的是矩阵运算单元。

以 MI50 为例：32 GB HBM2、标称带宽约 1 TB/s，二手价格低，手上已经有的也不必换掉。两张就能把
Flash-Next 的 24,576 个 expert 全部放进显存，decode 很强。但 gfx906 没有矩阵运算单元（没有 MFMA，
也没有 tensor core），两张卡 32K prefill 约 400 tok/s，改成分层 prefill 也只到 625 tok/s。
配上一张 prefill 强的 RTX 3070（约 1,500–1,650 tok/s），MI50 就只做 decode；以后要升级，也只需要换其中一边。
其他“decode 强 + prefill 强”的组合原理上同样适用，但目前没有实测。

### 引擎

参考实现用的是 Strata：它能把 session 保存下来再读回，而且同一份源码可以编译出 CUDA 和 HIP 两个后端，
session 能在两者之间交接。

## 工作原理

prefill 和 decode 对硬件的要求不同：prefill 是大批量矩阵运算，decode 主要受显存带宽限制。
PDHyperMix 让两种卡各做一段：

- **prefill 卡**：RTX 3070 跑 Strata 的 CUDA 版本，prefill 约 1,500–1,650 tok/s。
- **decode 卡**：2× MI50 跑 Strata 的 HIP（gfx906）版本，全部 expert 都在显存里。
- **session 交接**：prefill 完成后，3070 把 session（KV 和 recurrent state）保存成 session 文件，
  MI50 加载后接着 decode。32K prompt 的 session 约 726 MB，经由内存交接约 2.2 秒。

![请求流程](assets/pd-flow.svg)

PD 分离通常用在数据中心（多节点、RDMA 互联的集群）。要在单机上做到，关键是这三点：

1. **expert 在内存里只放一份。** IQ3_S 的 expert 约 47 GiB，两个引擎各加载一份
   会超过 94 GB 内存。Strata 的 `--shared-expert-arena` 把它放进 `/dev/shm`
   的共享文件（共享 expert 区），两个引擎共用。
2. **两个引擎出自同一份源码。** session 带有配置指纹，两边的版本、KV 格式、
   `--kv-resident` 必须一致，session 交接才会成功。
3. **session 交接走内存。** Strata 保存 session 时用 `O_DIRECT` 加 `fsync`，放在 SATA SSD
   上 32K 存取共 4.8 秒，放在 `/dev/shm` 是 2.2 秒。

## 内存与显存分配

![内存与显存分配](assets/memory-map.svg)

- **RTX 3070**：dense 层、MTP、262K 的完整 KV，以及 2,980 个 expert；其余 expert 在 prefill 时从内存流式读取。
- **MI50 #0**：dense 层、KV、MTP，以及 11,771 个 expert。
- **MI50 #1**：12,805 个 expert。两张合计放下全部 24,576 个，decode 时不从内存读 expert。
- **内存**：共享 expert 区 46.8 GiB、两个引擎和系统约 7 GiB、session 文件。

## decode 端的调整

长上下文时 decode 会变慢，原始配置在 258K 是 30 tok/s。做了两项调整：

- **KV 全部放显存**（`--kv-resident 262144`）：262K 的 KV 为 3.1 GiB，被挤出来的 expert
  改放到 MI50 #1。各个上下文长度都快了 4–14%。
- **gfx906 的 indexer top-k 改为两段式**（fork，`STRATA_GFX906_TOPK_SPLIT=1`）：原本每个
  query 只用一个 256 线程的 CTA，258K 时每个 window 占 12 ms。改写后结果逐位一致，
  这一步快了 6 倍。

![decode 速度随上下文深度](assets/decay.svg)

## 实测数据

Strata 本身的 prefill 速度几乎不随上下文长度衰减。在这个架构下（3070 prefill、2× MI50 decode），
实测如下：

| prompt 长度 | prefill（tok/s） | decode（tok/s） | TTFT（prefill + session 交接） |
|---|---|---|---|
| 16K | 1,533 | 43.8 | 12.2 s |
| 32K | 1,597 | 46.5 | 22.5 s |
| 64K | 1,681 | 40.0 | 42.4 s |
| 128K | 1,643 | 40.5 | 85.6 s |
| 192K | 1,589 | 39.3 | 132.6 s |
| 258K | 1,543 | 39.4 | 179.8 s |

decode 按任务类型（短上下文）：日常对话 47.3、编程 60.6、数学 60.6 tok/s。

上表为正式配置（KV 全放显存、两段式 top-k），decode 每次生成 300 token；TTFT 不含 decode 端加载 session 后补算几个 token 的时间（约 0.1–0.3 s）。

对照同引擎、同量化的社区实测，和单张 R9700 32 GB 相比：prefill 约快 2.1–2.3 倍（32K／128K），
decode 约为它的一半（46.5 对 95.6 tok/s）。老架构真正的痛点在 prefill；decode 保持在 40–50 tok/s，
日常使用已经够用（[对照表](docs/BENCHMARKS.zh-CN.md#与其他硬件的单请求对照)）。测试方法和完整数据见 [BENCHMARKS](docs/BENCHMARKS.zh-CN.md)。

## 硬件

| 组件 | 规格 |
|---|---|
| decode 卡 | 2× AMD Instinct MI50 32 GB（gfx906），sclk 上限 1,339 MHz、功耗上限 170 W |
| prefill 卡 | NVIDIA RTX 3070 16 GB（魔改版，另解锁 FP16/BF16，见 [BENCHMARKS](docs/BENCHMARKS.zh-CN.md#prefill-卡)） |
| CPU / 主板 | AMD Threadripper 1920X，ASUS PRIME X399-A，PCIe 3.0 |
| 内存 | 94 GB |
| 存储 | SATA SSD |
| 软件 | Ubuntu 22.04、kernel 6.8、ROCm（HIP）7.14、NVIDIA 驱动 580 / CUDA 13 |

## 测试过的组合

| decode | prefill | 状态 |
|---|---|---|
| 2× MI50 32 GB | RTX 3070 16 GB（魔改、已解锁） | 实测，本页数据 |

其他组合应该也能跑，但我们没有测试过，例如 prefill 换成 RTX 20／40／50（CUDA 镜像默认会编译这些 arch），
或 decode 换成 MI60／Radeon VII。如果你跑了别的组合，欢迎提 issue 贴上 `bench_tasks.py` 和 `bench_depth.py`
的输出，或者直接提 PR 把结果加进这张表。

## 需求

- **内存**：IQ3_S 约需 72 GB 以上（共享 expert 区 47 GiB、两个引擎约 15 GB、系统约 4 GB、session 文件最多约 4 GB）。
- **prefill 卡**：IQ3_S、262K 上下文时，dense、KV、MTP 和缓冲区约 9.5 GiB，建议 12 GB 以上。
- **decode 卡**：显存要放得下全部 expert；2× MI50 32 GB 刚好放得下 IQ3_S。

上面的内存和显存门槛是估算值，没有实测。

## 快速开始

```bash
git clone -b gfx906-ep https://github.com/MarcoTool/Strata.git strata-src   # 含 gfx906 top-k 的 Strata 分支
cd scripts
cp pd.env.example pd.env       # 配置 Strata checkout、路径、GPU 编号
./pd_build.sh                  # 用这份 checkout 编译出两个引擎镜像（CUDA、gfx906）
./pd_up.sh                     # 启动 prefill 引擎、decode 引擎和路由
# OpenAI 兼容端点：http://127.0.0.1:20030/v1
./pd_down.sh --free-arena      # 停止并释放共享 expert 区
```

我们不提供预编译镜像：`pd_build.sh` 会用 `docker/Dockerfile.cuda` 和 `docker/Dockerfile.gfx906`
在你的机器上构建。完整步骤见 [DEPLOY](docs/DEPLOY.zh-CN.md)。

## 已知问题

- 每个引擎一次处理一个请求（session 存取和并行请求模式互斥）。同时两个请求时整体只快 16%。
- 输出不保证逐字可复现：prefill（CUDA）和 decode（HIP）的浮点舍入不同。
- 另外运行任何 Strata 进程时，必须使用同一个 `--shared-expert-arena`，否则会多占约 47 GB 内存。
- 短上下文的日常对话 decode 约 47 tok/s；编程和数学题约 60 tok/s。

完整列表和排查方法见 [LESSONS](docs/LESSONS.zh-CN.md)。

## 目录

```
assets/    图（SVG）
docker/    Dockerfile.cuda（prefill 引擎）、Dockerfile.gfx906（decode 引擎）
docs/      DEPLOY、BENCHMARKS、LESSONS（*.zh-CN.md 为简体中文版）
scripts/   路由、引擎配置、编译、启动和测试脚本
```

## 许可证与致谢

本仓库以 MIT 许可证发布（见 [LICENSE](LICENSE)）。推理引擎为 [Strata](https://github.com/Niko1221/Strata)（MIT，© Niko1221 and the Strata contributors）。
decode 端的 top-k 改动目前在我们的 fork 分支，之后会向上游提 PR。
