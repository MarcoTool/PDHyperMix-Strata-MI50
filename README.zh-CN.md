# PDHyperMix-Strata-MI50

[English](README.md) · [繁體中文](README.zh-TW.md) · **简体中文**

**在单台主机上实现 prefill / decode 分离：由高显存带宽的 GPU 负责 decode，由矩阵算力强的 GPU 负责 prefill。**

[工作原理](#工作原理) · [实测数据](docs/BENCHMARKS.zh-CN.md) · [部署](docs/DEPLOY.zh-CN.md) · [已知问题](docs/LESSONS.zh-CN.md)

## 概述

PD 分离通常部署在多节点、通过 RDMA 互联的数据中心集群。本项目在一台主机上实现同样的架构，参考配置为 2× AMD Instinct MI50 搭配一张 RTX 3070，推理引擎为 [Strata](https://github.com/Niko1221/Strata)，模型为 Qwen3.8-Flash-Next IQ3_S。

| 指标 | 只用 MI50 | PDHyperMix |
|---|---|---|
| prefill，32K prompt | 约 400 tok/s | **约 1,600 tok/s** |
| decode，64K–258K | 30–35 tok/s | **39–40 tok/s** |
| TTFT，32K prompt | 约 80 s | **约 24 s** |
| 32K prompt 加 200 token 回答的总耗时 | 85.2 s | **27.8 s** |

本项目包含：

- PD 分离的架构与内存分配
- 可直接使用的 Dockerfile、配置文件与启动脚本
- Strata 的 gfx906 decode 改动：两段式 indexer top-k，位于 fork 分支 [`gfx906-ep`](https://github.com/MarcoTool/Strata/tree/gfx906-ep)，将向上游提交 PR

目前仅在上述配置上验证。更换硬件所需的调整见 [DEPLOY](docs/DEPLOY.zh-CN.md)。

## 设计

### GPU 搭配

prefill 与 decode 对硬件的需求不同。prefill 是大批量矩阵运算，取决于矩阵运算单元；decode 主要受显存带宽与显存容量限制。将两个阶段拆开后，可以分别选用最合适的 GPU。

MI50 拥有 32 GB HBM2 与约 1 TB/s 的标称带宽，两张即可将 Flash-Next 的 24,576 个 expert 全部放入显存。gfx906 没有 MFMA 等矩阵运算单元，两张卡的 32K prefill 约 400 tok/s，改用分层 prefill 也仅 625 tok/s。搭配 RTX 3070 负责 prefill 后，prefill 提升至 1,500–1,650 tok/s，MI50 只负责 decode。两端可以各自升级。

其他“decode 强、prefill 强”的组合原理上同样适用，尚未实测。

### 引擎

参考实现使用 Strata。Strata 支持 session 的保存与加载，且同一份源码可编译为 CUDA 与 HIP 两个后端，session 因此能在两个后端之间交接。

### fork 分支内容

fork 分支 [`gfx906-ep`](https://github.com/MarcoTool/Strata/tree/gfx906-ep) 以上游 Strata 0.1.43.1 为基础，包含下列 commit。本页所有数据均基于此分支的代码实测。

| commit | 内容 | PDHyperMix 是否使用 |
|---|---|---|
| [`6d07261`](https://github.com/MarcoTool/Strata/commit/6d072612) | gfx906 两段式 indexer top-k，由 `STRATA_GFX906_TOPK_SPLIT=1` 启用 | 使用 |
| [`2d8a158`](https://github.com/MarcoTool/Strata/commit/2d8a1582) | gfx906 分层 prefill 原型，由 `STRATA_GFX906_SPLIT_PREFILL` 启用 | 未使用；默认关闭，不影响运行结果 |
| [`c94d134`](https://github.com/MarcoTool/Strata/commit/c94d134f) | 上述组件的英文说明文档 | 仅文档，不涉及代码 |

分层 prefill 原型是在改用 PD 分离之前，让 MI50 自行加速 prefill 的实验，保留在分支中，使分支内容与实测版本一致。

## 工作原理

- **prefill**：RTX 3070 运行 Strata 的 CUDA 版本。
- **decode**：2× MI50 运行 Strata 的 HIP gfx906 版本，全部 expert 常驻显存。
- **session 交接**：prefill 完成后，3070 将 KV 与 recurrent state 保存为 session 文件，MI50 加载后继续 decode。32K prompt 的 session 约 726 MB，经内存交接约需 2.2 秒。

![请求流程](assets/pd-flow.svg)

在单机上实现 PD 分离有三个关键：

1. **expert 在内存中只保留一份。** IQ3_S 的 expert 约 47 GiB，两个引擎各加载一份会超过 94 GB 内存。Strata 的 `--shared-expert-arena` 将 expert 放在 `/dev/shm` 的共享文件中，由两个引擎共同映射。
2. **两个引擎使用同一份源码。** session 带有配置指纹，两端的版本、KV 格式与 `--kv-resident` 必须一致，交接才会成功。
3. **session 经由内存交接。** Strata 以 `O_DIRECT` 与 `fsync` 写入 session。32K session 的存取在 SATA SSD 上需 4.8 秒，在 `/dev/shm` 上为 2.2 秒。

## 内存与显存分配

![内存与显存分配](assets/memory-map.svg)

| 设备 | 内容 |
|---|---|
| RTX 3070 | dense 层、MTP、262K 完整 KV、2,980 个 expert；其余 expert 在 prefill 时从内存流式读取 |
| MI50 #0 | dense 层、KV、MTP、11,771 个 expert |
| MI50 #1 | 12,805 个 expert；两张合计覆盖全部 24,576 个，decode 无需从内存读取 expert |
| 内存 | 共享 expert 区 46.8 GiB、两个引擎与系统约 7 GiB、session 文件 |

## decode 端优化

原始配置下，decode 在 258K 上下文降至 30 tok/s。本项目做了两项调整：

- **KV 全部放入显存。** 设置 `--kv-resident 262144`，262K 的 KV 为 3.1 GiB，被挤出的 expert 移至 MI50 #1。各上下文长度提升 4–14%。
- **gfx906 indexer top-k 改为两段式。** 由 `STRATA_GFX906_TOPK_SPLIT=1` 启用，需要 fork 分支。原实现每个 query 仅使用一个 256 线程的 CTA，258K 时每个 window 耗时 12 ms；改写后结果逐位一致，该步骤加速 6 倍。

![decode 速度与上下文深度](assets/decay.svg)

## 实测数据

Strata 的 prefill 速度几乎不随上下文长度衰减。本配置的实测结果：

| prompt 长度 | prefill tok/s | decode tok/s | TTFT |
|---|---|---|---|
| 16K | 1,533 | 43.8 | 12.2 s |
| 32K | 1,597 | 46.5 | 22.5 s |
| 64K | 1,681 | 40.0 | 42.4 s |
| 128K | 1,643 | 40.5 | 85.6 s |
| 192K | 1,589 | 39.3 | 132.6 s |
| 258K | 1,543 | 39.4 | 179.8 s |

测试条件：KV 全部放入显存、启用两段式 top-k，每次生成 300 token。TTFT 为 prefill 加上 session 交接的时间，不含 decode 端加载后补算数个 token 的 0.1–0.3 s。

短上下文下按任务类型的 decode 速度：日常对话 47.3 tok/s，编程 60.6 tok/s，数学 60.6 tok/s。

与同引擎、同量化的社区实测相比，相对于单张 R9700 32 GB，本配置的 prefill 快 2.1–2.3 倍，decode 约为其一半，46.5 对 95.6 tok/s。完整对照与测试方法见 [BENCHMARKS](docs/BENCHMARKS.zh-CN.md#与其他硬件的单请求对照)。

## 测试硬件

| 组件 | 规格 |
|---|---|
| decode | 2× AMD Instinct MI50 32 GB，gfx906，sclk 上限 1,339 MHz，功耗上限 170 W |
| prefill | NVIDIA RTX 3070 16 GB 魔改版，已解锁 FP16/BF16，详见 [BENCHMARKS](docs/BENCHMARKS.zh-CN.md#prefill-卡) |
| CPU / 主板 | AMD Threadripper 1920X，ASUS PRIME X399-A，PCIe 3.0 |
| 内存 | 94 GB |
| 存储 | SATA SSD |
| 软件 | Ubuntu 22.04，kernel 6.8，ROCm 7.14，NVIDIA 驱动 580，CUDA 13 |

## 已验证组合

| decode | prefill | 状态 |
|---|---|---|
| 2× MI50 32 GB | RTX 3070 16 GB | 已实测，即本页数据 |

其他组合尚未验证，例如以 RTX 20／40／50 系列负责 prefill，或以 MI60、Radeon VII 负责 decode；CUDA 镜像默认已包含这些 NVIDIA 架构。欢迎通过 issue 反馈 `bench_tasks.py` 与 `bench_depth.py` 的结果，或提交 PR 将结果加入此表。

## 系统要求

| 项目 | 要求 |
|---|---|
| 内存 | IQ3_S 约 72 GB 以上：共享 expert 区 47 GiB、两个引擎约 15 GB、系统约 4 GB、session 文件最多约 4 GB |
| prefill GPU | IQ3_S、262K 上下文时，dense、KV、MTP 与缓冲区约 9.5 GiB，建议 12 GB 以上 |
| decode GPU | 显存需容纳全部 expert；2× MI50 32 GB 恰好容纳 IQ3_S |

以上门槛为估算值，未经实测。

## 快速开始

```bash
git clone https://github.com/MarcoTool/PDHyperMix-Strata-MI50.git
git clone -b gfx906-ep https://github.com/MarcoTool/Strata.git strata-src

cd PDHyperMix-Strata-MI50/scripts
cp pd.env.example pd.env        # 将 STRATA_SRC 指向 strata-src，并配置数据路径与 GPU 编号
./pd_build.sh                   # 以同一份源码构建 prefill 与 decode 两个镜像
./pd_up.sh                      # 启动 prefill 引擎、decode 引擎与路由
./pd_down.sh --free-arena       # 停止服务并释放共享 expert 区
```

服务启动后提供 OpenAI 兼容端点 `http://127.0.0.1:20030/v1`。本项目不提供预构建镜像，`pd_build.sh` 会依据 `docker/Dockerfile.cuda` 与 `docker/Dockerfile.gfx906` 在本机构建。模型下载与完整步骤见 [DEPLOY](docs/DEPLOY.zh-CN.md)。

## 已知限制

- 每个引擎同时只处理一个请求，session 存取与并行请求模式互斥。两个请求同时进入时，整体仅缩短 16%。
- 输出不保证逐字可复现，CUDA 与 HIP 的浮点舍入不同。
- PD 运行期间另外启动的 Strata 进程必须使用同一个 `--shared-expert-arena`，否则会再分配约 47 GB 内存。

完整列表与排查方法见 [LESSONS](docs/LESSONS.zh-CN.md)。

## 目录结构

```
assets/    架构图
docker/    prefill 与 decode 引擎的 Dockerfile
docs/      部署、实测数据、已知问题；*.zh-CN.md 为简体中文版
scripts/   路由、引擎配置、构建、启动与测试脚本
```

## 许可证与致谢

本项目以 MIT 许可证发布，详见 [LICENSE](LICENSE)。推理引擎为 [Strata](https://github.com/Niko1221/Strata)，MIT 许可证，© Niko1221 and the Strata contributors。
