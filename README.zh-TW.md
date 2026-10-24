# PDHyperMix-Strata-MI50

[English](README.md) · **繁體中文** · [简体中文](README.zh-CN.md)

**在一台家用機上做 prefill / decode 分流：高記憶體頻寬的卡負責 decode，矩陣運算強的卡負責 prefill。**

[運作方式](#運作方式) · [實測數字](docs/BENCHMARKS.md) · [部署](docs/DEPLOY.md) · [已知問題](docs/LESSONS.md)

2× MI50 + RTX 3070，Qwen3.8-Flash-Next IQ3_S，引擎是 [Strata](https://github.com/Niko1221/Strata)：

| | 只用 MI50 | PDHyperMix |
|---|---|---|
| prefill，32K prompt | 約 400 tok/s | **約 1,600 tok/s** |
| decode，64K–258K | 30–35 tok/s | **39–40 tok/s** |
| TTFT（time to first token），32K prompt | 約 80 s | **約 24 s** |
| 32K prompt + 200 token 回答的總時間 | 85.2 s | **27.8 s** |

這個 repo 包含三樣東西：PD 分流的做法、可以照著跑的設定與腳本，以及一項 Strata 的 gfx906 改動
（decode 端的兩段式 top-k，目前在我們的 [fork 分支 `gfx906-ep`](https://github.com/MarcoTool/Strata/tree/gfx906-ep)，之後會送 PR 給上游）。
目前只在我們這台機器上測過，換硬體時需要的調整寫在 [DEPLOY](docs/DEPLOY.md)。

### 配對方式

prefill 和 decode 拆開之後，可以把一張 decode 強的卡和一張 prefill 強的卡配在一起，各做擅長的那一段。
decode 強，看的是記憶體頻寬和 VRAM 容量；prefill 強，看的是矩陣運算單元。

以 MI50 為例：32 GB HBM2、規格頻寬約 1 TB/s，二手價格低，手上已經有的人也不必換掉。兩張就能把
Flash-Next 的 24,576 個 expert 全部放進 VRAM，decode 很強。但 gfx906 沒有矩陣運算單元（沒有 MFMA，
也沒有 tensor core），兩張卡 32K prefill 約 400 tok/s，改成分層 prefill 也只到 625 tok/s。
配上一張 prefill 強的 RTX 3070（約 1,500–1,650 tok/s），MI50 就只做 decode；之後要升級，也只需要換其中一邊。
其他「decode 強 + prefill 強」的組合在原理上也適用，但目前沒有實測。

### 引擎

參考實作用的是 Strata：它能把 session 存出來再讀回，而且同一份原始碼可以編成 CUDA 和 HIP 兩個後端，
session 能在兩者之間交接。

## 運作方式

prefill 和 decode 對硬體的要求不同：prefill 是大批矩陣運算，decode 主要受記憶體頻寬限制。
PDHyperMix 讓兩種卡各做一段：

- **prefill 卡**：RTX 3070 跑 Strata 的 CUDA build，prefill 約 1,500–1,650 tok/s。
- **decode 卡**：2× MI50 跑 Strata 的 HIP（gfx906）build，全部 expert 都在 VRAM。
- **session 交接**：prefill 完成後，3070 把 session（KV 與 recurrent state）存成 session 檔，
  MI50 載入後接著 decode。32K prompt 的 session 約 726 MB，經由 RAM 交接約 2.2 秒。

![請求流程](assets/pd-flow.svg)

PD 分流通常用在資料中心（多節點、以 RDMA 互連的叢集）。要在單機上做到，
關鍵是這三件事：

1. **expert 在 RAM 裡只放一份。** IQ3_S 的 expert 約 47 GiB，兩個引擎各載一份
   會超過 94 GB 的 RAM。Strata 的 `--shared-expert-arena` 把它放進 `/dev/shm`
   的共享檔（共用 expert 區），兩個引擎共用。
2. **兩個引擎出自同一份原始碼。** session 帶有設定指紋，兩邊的版本、KV 格式、
   `--kv-resident` 必須一致，session 交接才會成功。
3. **session 交接走 RAM。** Strata 存 session 時用 `O_DIRECT` 加 `fsync`，放在 SATA SSD
   上 32K 存讀共 4.8 秒，放在 `/dev/shm` 是 2.2 秒。

## 記憶體配置

![記憶體配置](assets/memory-map.svg)

- **RTX 3070**：dense 層、MTP、262K 的完整 KV，以及 2,980 個 expert；其餘 expert 在 prefill 時從 RAM 串流。
- **MI50 #0**：dense 層、KV、MTP，以及 11,771 個 expert。
- **MI50 #1**：12,805 個 expert。兩張合計放下全部 24,576 個，decode 時不從 RAM 讀 expert。
- **RAM**：共用 expert 區 46.8 GiB、兩個引擎與系統約 7 GiB、session 檔。

## decode 端的調整

長上下文時 decode 會變慢，原始設定在 258K 是 30 tok/s。做了兩項調整：

- **KV 全部放 VRAM**（`--kv-resident 262144`）：262K 的 KV 為 3.1 GiB，被擠出來的 expert
  改放到 MI50 #1。各個上下文長度都快了 4–14%。
- **gfx906 的 indexer top-k 改為兩段式**（fork，`STRATA_GFX906_TOPK_SPLIT=1`）：原本每個
  query 只用一個 256 執行緒的 CTA，258K 時每個 window 佔 12 ms。改寫後結果逐位元相同，
  這一步快了 6 倍。

![decode 速度隨上下文深度](assets/decay.svg)

## 實測數字

Strata 本身的 prefill 速度幾乎不隨上下文長度衰退。在這個架構下（3070 prefill、2× MI50 decode），
實測如下：

| prompt 長度 | prefill（tok/s） | decode（tok/s） | TTFT（prefill + session 交接） |
|---|---|---|---|
| 16K | 1,533 | 43.8 | 12.2 s |
| 32K | 1,597 | 46.5 | 22.5 s |
| 64K | 1,681 | 40.0 | 42.4 s |
| 128K | 1,643 | 40.5 | 85.6 s |
| 192K | 1,589 | 39.3 | 132.6 s |
| 258K | 1,543 | 39.4 | 179.8 s |

decode 依任務類型（短上下文）：一般對話 47.3、程式 60.6、數學 60.6 tok/s。

上表為正式設定（KV 全放 VRAM、兩段式 top-k），decode 每次生成 300 token；TTFT 不含 decode 端讀入 session 後補讀幾個 token 的時間（約 0.1–0.3 s）。

對照同引擎、同量化的社群實測，跟單張 R9700 32 GB 相比：prefill 約快 2.1–2.3 倍（32K／128K），
decode 約為它的一半（46.5 對 95.6 tok/s）。舊架構真正的痛點在 prefill；decode 維持在 40–50 tok/s，
日常使用已經夠用（[對照表](docs/BENCHMARKS.md#與其他硬體的單請求對照)）。量測方法與完整數據見 [BENCHMARKS](docs/BENCHMARKS.md)。

## 硬體

| 元件 | 規格 |
|---|---|
| decode 卡 | 2× AMD Instinct MI50 32 GB（gfx906），sclk 上限 1,339 MHz、功耗上限 170 W |
| prefill 卡 | NVIDIA RTX 3070 16 GB（改裝版，另解鎖 FP16/BF16，見 [BENCHMARKS](docs/BENCHMARKS.md#prefill-卡)） |
| CPU / 主機板 | AMD Threadripper 1920X，ASUS PRIME X399-A，PCIe 3.0 |
| RAM | 94 GB |
| 儲存 | SATA SSD |
| 軟體 | Ubuntu 22.04、kernel 6.8、ROCm（HIP）7.14、NVIDIA 驅動 580 / CUDA 13 |

## 測過的組合

| decode | prefill | 狀態 |
|---|---|---|
| 2× MI50 32 GB | RTX 3070 16 GB（改裝、已解鎖） | 實測，本頁數字 |

其他組合應該也能跑，但我們沒有測過，例如 prefill 換成 RTX 20／40／50（CUDA 映像預設會編進這些 arch），
或 decode 換成 MI60／Radeon VII。如果你跑了別的組合，歡迎開 issue 貼上 `bench_tasks.py` 和 `bench_depth.py`
的輸出，或直接送 PR 把結果加進這張表。

## 需求

- **RAM**：IQ3_S 約需 72 GB 以上（共用 expert 區 47 GiB、兩個引擎約 15 GB、系統約 4 GB、session 檔最多約 4 GB）。
- **prefill 卡**：IQ3_S、262K 上下文時，dense、KV、MTP 與緩衝區約 9.5 GiB，建議 12 GB 以上。
- **decode 卡**：VRAM 要放得下全部 expert；2× MI50 32 GB 剛好放得下 IQ3_S。

上面的 RAM 和 VRAM 門檻是推估值，沒有實測。

## 快速開始

```bash
git clone -b gfx906-ep https://github.com/MarcoTool/Strata.git strata-src   # 含 gfx906 top-k 的 Strata 分支
cd scripts
cp pd.env.example pd.env       # 設定 Strata checkout、路徑、GPU 編號
./pd_build.sh                  # 用那份 checkout 編出兩個引擎映像（CUDA、gfx906）
./pd_up.sh                     # 啟動 prefill 引擎、decode 引擎與路由
# OpenAI 相容端點：http://127.0.0.1:20030/v1
./pd_down.sh --free-arena      # 停止並釋放共用 expert 區
```

我們不提供預先編好的映像：`pd_build.sh` 會用 `docker/Dockerfile.cuda` 和 `docker/Dockerfile.gfx906`
在你的機器上 build。完整步驟見 [DEPLOY](docs/DEPLOY.md)。

## 已知問題

- 每個引擎一次處理一個請求（session 存讀與平行請求模式互斥）。同時兩個請求時整體只快 16%。
- 輸出不保證逐字可重現：prefill（CUDA）與 decode（HIP）的浮點捨入不同。
- 另外執行任何 Strata 行程時，必須使用同一個 `--shared-expert-arena`，否則會多配約 47 GB RAM。
- 短上下文的一般對話 decode 約 47 tok/s；程式與數學題約 60 tok/s。

完整清單與排解方式見 [LESSONS](docs/LESSONS.md)。

## 目錄

```
assets/    圖（SVG）
docker/    Dockerfile.cuda（prefill 引擎）、Dockerfile.gfx906（decode 引擎）
docs/      DEPLOY、BENCHMARKS、LESSONS
scripts/   路由、引擎設定、編譯、啟動與量測腳本
```

## 授權與致謝

本 repo 以 MIT 授權釋出（見 [LICENSE](LICENSE)）。推論引擎為 [Strata](https://github.com/Niko1221/Strata)（MIT，© Niko1221 and the Strata contributors）。
decode 端的 top-k 改動目前在我們的 fork 分支，之後會送 PR 給上游。
