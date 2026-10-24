# 部署步驟

這份步驟對應 README 的配置：2× MI50 負責 decode、一張 NVIDIA 卡負責 prefill、Strata 跑 Flash-Next IQ3_S。換成別的卡或量化時，要調整的地方都標了出來。

## 0. 先確認放得下

| 項目 | 需要 | 我們的機器 |
|---|---|---|
| RAM | expert 大小 + 約 15 GB（兩個引擎）+ 約 4 GB（系統）+ session 檔（最多約 4 GB） | 94 GB，IQ3_S 的 expert 47 GiB |
| prefill 卡 VRAM | dense + KV + MTP + 緩衝區，IQ3_S、262K 時約 9.5 GiB | 3070 16 GB |
| decode 端 VRAM | 全部 expert + dense + KV + MTP | 2× MI50 32 GB，剛好放下全部 24,576 個 expert |

`/dev/shm` 要放得下共用 expert 區、一份 session 檔，再加上 Strata 要求的 4 GiB 保留空間：

```bash
sudo mount -o remount,size=64G /dev/shm     # 重開機後會恢復；要永久生效就寫進 /etc/fstab
```

## 1. 取得 Strata 原始碼、填設定

兩個引擎必須是**同一個 commit**，否則 session 的設定指紋對不上，交接會失敗（[LESSONS.md](LESSONS.md#3-session-指紋兩個引擎的設定必須一模一樣)）。所以兩個映像都從同一份 checkout 編。

```bash
git clone -b gfx906-ep https://github.com/MarcoTool/Strata.git strata-src     # 含兩段式 top-k 的分支；上游 PR 合併後可改用上游
cd PDHyperMix-Strata-MI50/scripts && cp pd.env.example pd.env && $EDITOR pd.env
```

至少要改的：`STRATA_SRC`（上面那份 checkout）、`DATA`、`WORK`，以及 GPU 編號。

模型（native pack、GGUF、MTP）照 Strata 的 setup 下載與轉換，放進 `DATA` 目錄。

## 2. 編譯兩個映像

```bash
./pd_build.sh
```

它用同一份 checkout 編出兩個映像，並把 commit 寫進映像裡；`pd_up.sh` 啟動前會比對，兩邊不同就拒絕啟動。

| 映像 | Dockerfile | 內容 |
|---|---|---|
| `IMG_PRE`（預設 `pdhypermix-pre`） | `docker/Dockerfile.cuda` | CUDA 13 的 prefill 引擎。預設一次編進 `75;80;86;89;120`（RTX 20／A 系列／30／40／50），執行時挑對應的那份 |
| `IMG_DEC`（預設 `pdhypermix-dec`） | `docker/Dockerfile.gfx906` | gfx906 的 decode 引擎。以 Strata 文件建議的社群映像 `mixa3607/rocm-gfx906`（HIP 7.14）為底，AMD 現在的 ROCm 已經不附 gfx906 的函式庫 |

- 只有一種 NVIDIA 卡的話，把 `CUDA_ARCHITECTURES` 改成自己的 arch（3070 是 `86`），編譯快很多。
- CPU 端的程式碼是針對編譯那台機器的 CPU 編的，請在要跑的那台機器上 build。
- 編譯時每個 nvcc／hipcc 工作會吃幾 GB 的 RAM，RAM 吃緊時把 `BUILD_JOBS` 調低。
- 宿主機只需要驅動：NVIDIA 驅動 580 以上加上 nvidia-container-toolkit；AMD 用 kernel 內建的 amdgpu 驅動。

## 3. 啟動

```bash
./pd_up.sh
```

依序執行的內容：

1. 產生兩個引擎的設定檔。
2. 確認兩個映像出自同一個 commit，並檢查 `/dev/shm` 的大小和可用 RAM，不夠就拒絕啟動。
3. 啟動 prefill 引擎。它會建立共用 expert 區，第一次約 2.5 分鐘。
4. 啟動 decode 引擎，綁同一個共用 expert 區。
5. 啟動路由。

完成後，客戶端打 `http://127.0.0.1:20030/v1`（OpenAI 相容）。

## 4. 驗證

```bash
python bench_tasks.py http://127.0.0.1:20062 mine          # 只量 decode 端，依任務類型
```

`bench_depth.py` 量 prefill、交接、decode 隨深度的變化。它要用 Strata 的 tokenizer，在 prefill 映像裡跑最省事（映像裡有 Strata 和它的 Python 套件）：

```bash
docker run --rm --network host -v "$PWD":/s:ro -v <語料目錄>:/c:ro -v <DATA>:/data:ro pdhypermix-pre \
  -c 'cd /opt/strata && STRATA_SRC=/opt/strata TOKENIZER_DIR=/data/packs/iq3_s/tokenizer/ .venv/bin/python /s/bench_depth.py /c/corpus.txt 4096 32768 131072'
```

拿去跟 [BENCHMARKS.md](BENCHMARKS.md) 比。先看兩件事：

- 交接有沒有成功：decode 引擎的 log 要出現 `session restored`，而且「N reused + 7 read」裡的 N 接近 prompt 長度。
- 32K prompt 的 TTFT 有沒有大幅縮短。

## 5. 停止

```bash
./pd_down.sh                 # 保留共用 expert 區，下次啟動比較快
./pd_down.sh --free-arena    # 連共用 expert 區一起刪掉，把約 47 GiB 的 RAM 還給系統
```

**跑任何其他會載入模型的程式之前，請先 `--free-arena`**，或者讓那個程式也綁同一個共用 expert 區（[LESSONS.md](LESSONS.md#1-ram-是硬上限任何額外的-strata-行程都要綁共用-expert-區)）。

## 換成別的配置時

| 要換的 | 需要注意 |
|---|---|
| prefill 卡 | 預設的 `CUDA_ARCHITECTURES` 會編進 RTX 20–50 的 arch，改 `PRE_GPU` 即可；但我們只測過 3070。VRAM 小於約 10 GB 時，262K 的 KV 放不下，要把 `MAX_CONTEXT` 和 `KV_RESIDENT` 一起調小（兩邊同時改） |
| decode 端只有一張 MI50 | 拿掉 `--expert-cache-device1`，放不下的 expert 會改從 RAM 讀，decode 會變慢 |
| 更小的量化（IQ2_XS、Q2_0） | 共用 expert 區變小，RAM 門檻跟著降低；注意 Q2_0 在 gfx906 上沒有專屬的 kernel 路徑 |
