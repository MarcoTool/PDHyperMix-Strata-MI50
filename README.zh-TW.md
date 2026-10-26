# PDHyperMix-Strata-MI50

[English](README.md) · **繁體中文** · [简体中文](README.zh-CN.md)

**在單一主機上實現 prefill / decode 分離：由高記憶體頻寬的 GPU 負責 decode，由矩陣運算能力強的 GPU 負責 prefill。**

[運作方式](#運作方式) · [實測數據](docs/BENCHMARKS.md) · [部署](docs/DEPLOY.md) · [已知問題](docs/LESSONS.md)

## 概述

PD 分離通常部署在多節點、以 RDMA 互連的資料中心叢集。本專案在一台主機上實現同樣的架構，參考配置為 2× AMD Instinct MI50 搭配一張 RTX 3070，推論引擎為 [Strata](https://github.com/Niko1221/Strata)，模型為 Qwen3.8-Flash-Next IQ3_S。

| 指標 | 只用 MI50 | PDHyperMix |
|---|---|---|
| prefill，32K prompt | 約 400 tok/s | **約 1,600 tok/s** |
| decode，64K–258K | 30–35 tok/s | **39–40 tok/s** |
| TTFT，32K prompt | 約 80 s | **約 24 s** |
| 32K prompt 加 200 token 回答的總時間 | 85.2 s | **27.8 s** |

本專案包含：

- PD 分離的架構與記憶體配置
- 可直接使用的 Dockerfile、設定檔與啟動腳本
- Strata 的 gfx906 decode 改動：兩段式 indexer top-k，位於 fork 分支 [`gfx906-ep`](https://github.com/MarcoTool/Strata/tree/gfx906-ep)，將提交 PR 至上游

目前僅在上述配置驗證。更換硬體所需的調整見 [DEPLOY](docs/DEPLOY.md)。

## 設計

### GPU 配對

prefill 與 decode 對硬體的需求不同。prefill 為大批次矩陣運算，取決於矩陣運算單元；decode 主要受記憶體頻寬與 VRAM 容量限制。將兩個階段拆開後，可以分別選用最適合的 GPU。

MI50 具備 32 GB HBM2 與約 1 TB/s 的規格頻寬，兩張即可將 Flash-Next 的 24,576 個 expert 全數放入 VRAM。gfx906 沒有 MFMA 等矩陣運算單元，兩張卡的 32K prefill 約 400 tok/s，改用分層 prefill 也僅 625 tok/s。搭配 RTX 3070 負責 prefill 後，prefill 提升至 1,500–1,650 tok/s，MI50 只負責 decode。兩端可以各自升級。

其他「decode 強、prefill 強」的組合在原理上同樣適用，尚未實測。

### 引擎

參考實作使用 Strata。Strata 支援 session 的儲存與載入，且同一份原始碼可編譯為 CUDA 與 HIP 兩個後端，session 因此能在兩個後端之間交接。

### fork 分支內容

fork 分支 [`gfx906-ep`](https://github.com/MarcoTool/Strata/tree/gfx906-ep) 以上游 Strata 0.1.43.1 為基礎，包含下列 commit。本頁所有數據皆以此分支的程式碼實測。

| commit | 內容 | PDHyperMix 是否使用 |
|---|---|---|
| [`6d07261`](https://github.com/MarcoTool/Strata/commit/6d072612) | gfx906 兩段式 indexer top-k，由 `STRATA_GFX906_TOPK_SPLIT=1` 啟用 | 使用 |
| [`2d8a158`](https://github.com/MarcoTool/Strata/commit/2d8a1582) | gfx906 分層 prefill 原型，由 `STRATA_GFX906_SPLIT_PREFILL` 啟用 | 未使用；預設關閉，不影響執行結果 |
| [`c94d134`](https://github.com/MarcoTool/Strata/commit/c94d134f) | 上述元件的英文說明文件 | 僅文件，不涉及程式碼 |

分層 prefill 原型是在改採 PD 分離之前，讓 MI50 自行加速 prefill 的實驗，保留於分支中，使分支內容與實測版本一致。

## 運作方式

- **prefill**：RTX 3070 執行 Strata 的 CUDA 版本。
- **decode**：2× MI50 執行 Strata 的 HIP gfx906 版本，全部 expert 常駐 VRAM。
- **session 交接**：prefill 完成後，3070 將 KV 與 recurrent state 存成 session 檔，MI50 載入後繼續 decode。32K prompt 的 session 約 726 MB，經 RAM 交接約需 2.2 秒。

![請求流程](assets/pd-flow.svg)

在單機上實現 PD 分離有三個關鍵：

1. **expert 在 RAM 中只保留一份。** IQ3_S 的 expert 約 47 GiB，兩個引擎各載一份會超過 96 GB 的 RAM。Strata 的 `--shared-expert-arena` 將 expert 放在 `/dev/shm` 的共享檔案中，由兩個引擎共同映射。
2. **兩個引擎使用同一份原始碼。** session 帶有設定指紋，兩端的版本、KV 格式與 `--kv-resident` 必須一致，交接才會成功。
3. **session 經由 RAM 交接。** Strata 以 `O_DIRECT` 與 `fsync` 寫入 session。32K session 的存取在 SATA SSD 上需 4.8 秒，在 `/dev/shm` 上為 2.2 秒。

## 記憶體配置

![記憶體配置](assets/memory-map.svg)

| 裝置 | 內容 |
|---|---|
| RTX 3070 | dense 層、MTP、262K 完整 KV、2,980 個 expert；其餘 expert 於 prefill 時自 RAM 串流 |
| MI50 #0 | dense 層、KV、MTP、11,771 個 expert |
| MI50 #1 | 12,805 個 expert；兩張合計涵蓋全部 24,576 個，decode 不需自 RAM 讀取 expert |
| RAM | 共享 expert 區 46.8 GiB、兩個引擎與系統約 7 GiB、session 檔 |

## decode 端最佳化

原始設定下，decode 在 258K 上下文降至 30 tok/s。本專案做了兩項調整：

- **KV 全部放入 VRAM。** 設定 `--kv-resident 262144`，262K 的 KV 為 3.1 GiB，被擠出的 expert 移至 MI50 #1。各上下文長度提升 4–14%。
- **gfx906 indexer top-k 改為兩段式。** 由 `STRATA_GFX906_TOPK_SPLIT=1` 啟用，需要 fork 分支。原實作每個 query 僅使用一個 256 執行緒的 CTA，258K 時每個 window 耗時 12 ms；改寫後結果逐位元相同，該步驟加速 6 倍。

![decode 速度與上下文深度](assets/decay.svg)

## 實測數據

Strata 的 prefill 速度幾乎不隨上下文長度衰退。本配置的實測結果：

| prompt 長度 | prefill tok/s | decode tok/s | TTFT |
|---|---|---|---|
| 16K | 1,533 | 43.8 | 12.2 s |
| 32K | 1,597 | 46.5 | 22.5 s |
| 64K | 1,681 | 40.0 | 42.4 s |
| 128K | 1,643 | 40.5 | 85.6 s |
| 192K | 1,589 | 39.3 | 132.6 s |
| 258K | 1,543 | 39.4 | 179.8 s |

測試條件：KV 全部放入 VRAM、啟用兩段式 top-k，每次生成 300 token。TTFT 為 prefill 加上 session 交接的時間，不含 decode 端載入後補算數個 token 的 0.1–0.3 s。

短上下文下依任務類型的 decode 速度：一般對話 47.3 tok/s，程式 60.6 tok/s，數學 60.6 tok/s。

與同引擎、同量化的社群實測相比，相對於單張 R9700 32 GB，本配置的 prefill 快 2.1–2.3 倍，decode 約為其一半，46.5 對 95.6 tok/s。完整對照與測試方法見 [BENCHMARKS](docs/BENCHMARKS.md#與其他硬體的單請求對照)。

## 測試硬體

| 元件 | 規格 |
|---|---|
| decode | 2× AMD Instinct MI50 32 GB，gfx906，sclk 上限 1,339 MHz，功耗上限 170 W |
| prefill | NVIDIA RTX 3070 16 GB 改裝版，已解鎖 FP16/BF16，詳見 [BENCHMARKS](docs/BENCHMARKS.md#prefill-卡) |
| CPU / 主機板 | AMD Threadripper 1920X，ASUS PRIME X399-A，PCIe 3.0 |
| RAM | 96 GB |
| 儲存 | SATA SSD |
| 軟體 | Ubuntu 22.04，kernel 6.8，ROCm 7.14，NVIDIA 驅動 580，CUDA 13 |

## 已驗證組合

| decode | prefill | 狀態 |
|---|---|---|
| 2× MI50 32 GB | RTX 3070 16 GB | 已實測，即本頁數據 |

其他組合尚未驗證，例如以 RTX 20／40／50 系列負責 prefill，或以 MI60、Radeon VII 負責 decode；CUDA 映像預設已包含這些 NVIDIA 架構。歡迎以 issue 回報 `bench_tasks.py` 與 `bench_depth.py` 的結果，或提交 PR 將結果加入此表。

## 系統需求

| 項目 | 需求 |
|---|---|
| RAM | IQ3_S 約 72 GB 以上：共享 expert 區 47 GiB、兩個引擎約 15 GB、系統約 4 GB、session 檔最多約 4 GB |
| prefill GPU | IQ3_S、262K 上下文時，dense、KV、MTP 與緩衝區約 9.5 GiB，建議 12 GB 以上 |
| decode GPU | VRAM 需容納全部 expert；2× MI50 32 GB 恰好容納 IQ3_S |

以上門檻為估算值，未經實測。

## 快速開始

```bash
git clone https://github.com/MarcoTool/PDHyperMix-Strata-MI50.git
git clone -b gfx906-ep https://github.com/MarcoTool/Strata.git strata-src

cd PDHyperMix-Strata-MI50/scripts
cp pd.env.example pd.env        # 將 STRATA_SRC 指向 strata-src，並設定資料路徑與 GPU 編號
./pd_build.sh                   # 以同一份原始碼建置 prefill 與 decode 兩個映像
./pd_up.sh                      # 啟動 prefill 引擎、decode 引擎與路由
./pd_down.sh --free-arena       # 停止服務並釋放共享 expert 區
```

服務啟動後提供 OpenAI 相容端點 `http://127.0.0.1:20030/v1`。本專案不提供預先建置的映像，`pd_build.sh` 會依 `docker/Dockerfile.cuda` 與 `docker/Dockerfile.gfx906` 在本機建置。模型下載與完整步驟見 [DEPLOY](docs/DEPLOY.md)。

## 路由

路由依 `PD_MIN_CHARS` 決定請求要不要先交給 prefill 卡讀：新增的文字達到這個字元數時，先由 prefill 引擎讀 prompt 並交接 session；否則直接由 decode 引擎處理。預設為 `0`，所有純文字請求都先交給 prefill 卡。prefill 引擎會沿用前幾輪已讀過的 KV，只讀新增的部分，decode 引擎載入 session 後直接生成。

以 agent 的工作方式實測：16K 文件之後，每輪加入約 500 token 的工具結果。`PD_MIN_CHARS` 為 6000 時，這些短增量留在 decode 引擎讀取，每輪 5.4–12.2 s，平均 8.5 s；設為 `0` 後每輪 5.3–5.8 s，平均 5.5 s。

## 影像輸入

在 `pd.env` 設定 `MMPROJ` 後，兩個引擎都會啟用影像功能。兩個引擎都必須加上 `--vision`，因為它是 session 指紋的一部分，少了任何一邊交接就會失敗。

圖片也能像文字一樣交接，但有一個條件：兩個引擎必須用同一個編碼器、在同一張卡上編碼。引擎判斷能否沿用 session 時，是以圖片向量的雜湊辨識一張圖；每個引擎收到請求都會重新編碼圖片，GPU 與 CPU 編碼器的結果不會逐位元相同，雜湊對不上，decode 引擎就只能重讀整個 prompt。同一個 CUDA 編碼器在同一張卡上分兩個程序編碼同一張圖，結果逐位元相同。

| 設定 | 預設 | 作用 |
|---|---|---|
| `DEC_ENCODER` | `prefill-gpu` | decode 引擎的編碼器：`prefill-gpu` 在 prefill 卡上執行 prefill 映像的 CUDA 編碼器，`pd_up.sh` 會自動取出並掛進 decode 容器；`cpu` 則在 CPU 上執行 |
| `PD_IMAGES` | `prefill` | 帶圖片的對話：`prefill` 由 prefill 卡讀圖並交接；`decode` 直接送 decode 引擎自行讀圖，搭配 `DEC_ENCODER=cpu` 使用 |

實測：

| 情境 | 結果 |
|---|---|
| 1920×1080 截圖，1,045 token，`prefill-gpu` 加 `prefill` | 7.2 s；prefill 卡讀 4.1 s，decode 引擎沿用 1,038 token，回答 1.2 s |
| 同類截圖，CPU 編碼器讀圖 | 最長 66.8 s |
| 小圖，GPU 與 CPU 編碼器混用並交接 | 14.1 s，decode 引擎重讀 |

`prefill-gpu` 會在 prefill 卡上多佔約 1.2 GiB，`pd_cfg.py` 會自動加大 prefill 引擎的 `--vram-reserve-mib`。

## 已知限制

- 每個引擎同時只處理一個請求，session 存取與平行請求模式互斥。兩個請求同時進入時，整體僅縮短 16%。
- 輸出不保證逐字可重現，CUDA 與 HIP 的浮點捨入不同。
- PD 執行期間另外啟動的 Strata 行程必須使用同一個 `--shared-expert-arena`，否則會再配置約 47 GB RAM。

完整清單與排除方式見 [LESSONS](docs/LESSONS.md)。

## 目錄結構

```
assets/    架構圖
docker/    prefill 與 decode 引擎的 Dockerfile
docs/      部署、實測數據、已知問題
scripts/   路由、引擎設定、建置、啟動與測試腳本
```

## 授權與致謝

本專案以 MIT 授權釋出，詳見 [LICENSE](LICENSE)。推論引擎為 [Strata](https://github.com/Niko1221/Strata)，MIT 授權，© Niko1221 and the Strata contributors。
