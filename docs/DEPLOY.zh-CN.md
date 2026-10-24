# 部署步骤

这份步骤对应 README 的配置：2× MI50 负责 decode、一张 NVIDIA 卡负责 prefill、Strata 跑 Flash-Next IQ3_S。换成别的卡或量化时，要调整的地方都标了出来。

## 0. 先确认放得下

| 项目 | 需要 | 我们的机器 |
|---|---|---|
| RAM | expert 大小 + 约 15 GB（两个引擎）+ 约 4 GB（系统）+ session 文件（最多约 4 GB） | 96 GB，IQ3_S 的 expert 47 GiB |
| prefill 卡 VRAM | dense + KV + MTP + 缓冲区，IQ3_S、262K 时约 9.5 GiB | 3070 16 GB |
| decode 端 VRAM | 全部 expert + dense + KV + MTP | 2× MI50 32 GB，刚好放下全部 24,576 个 expert |

`/dev/shm` 要放得下共享 expert 区、一份 session 文件，再加上 Strata 要求的 4 GiB 保留空间：

```bash
sudo mount -o remount,size=64G /dev/shm     # 重启后会恢复；要永久生效就写进 /etc/fstab
```

## 1. 获取 Strata 源码、填写配置

两个引擎必须是**同一个 commit**，否则 session 的配置指纹对不上，交接会失败（[LESSONS.md](LESSONS.zh-CN.md#3-session-指纹两个引擎的配置必须完全一致)）。所以两个镜像都从同一份 checkout 编译。

```bash
git clone -b gfx906-ep https://github.com/MarcoTool/Strata.git strata-src     # 含两段式 top-k 的分支；上游 PR 合并后可改用上游
cd PDHyperMix-Strata-MI50/scripts && cp pd.env.example pd.env && $EDITOR pd.env
```

至少要改的：`STRATA_SRC`（上面那份 checkout）、`DATA`、`WORK`，以及 GPU 编号。

模型（native pack、GGUF、MTP）按 Strata 的 setup 下载和转换，放进 `DATA` 目录。

## 2. 编译两个镜像

```bash
./pd_build.sh
```

它用同一份 checkout 编译出两个镜像，并把 commit 写进镜像里；`pd_up.sh` 启动前会比对，两边不同就拒绝启动。

| 镜像 | Dockerfile | 内容 |
|---|---|---|
| `IMG_PRE`（默认 `pdhypermix-pre`） | `docker/Dockerfile.cuda` | CUDA 13 的 prefill 引擎。默认一次编译 `75;80;86;89;120`（RTX 20／A 系列／30／40／50），运行时选对应的那份 |
| `IMG_DEC`（默认 `pdhypermix-dec`） | `docker/Dockerfile.gfx906` | gfx906 的 decode 引擎。以 Strata 文档推荐的社区镜像 `mixa3607/rocm-gfx906`（HIP 7.14）为基础，AMD 现在的 ROCm 已经不附带 gfx906 的库 |

- 只有一种 NVIDIA 卡的话，把 `CUDA_ARCHITECTURES` 改成自己的 arch（3070 是 `86`），编译快很多。
- CPU 端的代码是针对编译那台机器的 CPU 编译的，请在要运行的那台机器上构建。
- 编译时每个 nvcc／hipcc 任务会占几 GB 内存，内存紧张时把 `BUILD_JOBS` 调低。
- 宿主机只需要驱动：NVIDIA 驱动 580 以上加 nvidia-container-toolkit；AMD 用内核自带的 amdgpu 驱动。

## 3. 启动

```bash
./pd_up.sh
```

依次执行的内容：

1. 生成两个引擎的配置文件。
2. 确认两个镜像出自同一个 commit，并检查 `/dev/shm` 的大小和可用 RAM，不够就拒绝启动。
3. 启动 prefill 引擎。它会建立共享 expert 区，第一次约 2.5 分钟。
4. 启动 decode 引擎，绑定同一个共享 expert 区。
5. 启动路由。

完成后，客户端访问 `http://127.0.0.1:20030/v1`（OpenAI 兼容）。

## 4. 验证

```bash
python bench_tasks.py http://127.0.0.1:20062 mine          # 只测 decode 端，按任务类型
```

`bench_depth.py` 测试 prefill、交接、decode 随深度的变化。它要用 Strata 的 tokenizer，在 prefill 镜像里运行最省事（镜像里有 Strata 和它的 Python 依赖）：

```bash
docker run --rm --network host -v "$PWD":/s:ro -v <语料目录>:/c:ro -v <DATA>:/data:ro pdhypermix-pre \
  -c 'cd /opt/strata && STRATA_SRC=/opt/strata TOKENIZER_DIR=/data/packs/iq3_s/tokenizer/ .venv/bin/python /s/bench_depth.py /c/corpus.txt 4096 32768 131072'
```

拿去跟 [BENCHMARKS.md](BENCHMARKS.zh-CN.md) 比。先看两件事：

- 交接有没有成功：decode 引擎的 log 要出现 `session restored`，而且“N reused + 7 read”里的 N 接近 prompt 长度。
- 32K prompt 的 TTFT 有没有大幅缩短。

## 5. 停止

```bash
./pd_down.sh                 # 保留共享 expert 区，下次启动比较快
./pd_down.sh --free-arena    # 连共享 expert 区一起删掉，把约 47 GiB 的 RAM 还给系统
```

**跑任何其他会加载模型的程序之前，请先 `--free-arena`**，或者让那个程序也绑定同一个共享 expert 区（[LESSONS.md](LESSONS.zh-CN.md#1-ram-是硬上限任何额外的-strata-进程都要绑定共享-expert-区)）。

## 换成别的配置时

| 要换的 | 需要注意 |
|---|---|
| prefill 卡 | 默认的 `CUDA_ARCHITECTURES` 会编译 RTX 20–50 的 arch，改 `PRE_GPU` 即可；但我们只测试过 3070。VRAM 小于约 10 GB 时，262K 的 KV 放不下，要把 `MAX_CONTEXT` 和 `KV_RESIDENT` 一起调小（两边同时改） |
| decode 端只有一张 MI50 | 拿掉 `--expert-cache-device1`，放不下的 expert 会改从 RAM 读，decode 会变慢 |
| 更小的量化（IQ2_XS、Q2_0） | 共享 expert 区变小，RAM 门槛跟着降低；注意 Q2_0 在 gfx906 上没有专属的 kernel 路径 |
