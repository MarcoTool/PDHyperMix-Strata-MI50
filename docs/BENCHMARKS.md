# 量測數字與條件

全部數字都在同一台機器上、2026-10-23 到 10-24 量測。量測時機器上沒有其他推論負載，每一項都註明了方法。沒有實測的推估，會標明「推估」。

## 共同條件

- **模型**：Qwen3.8-Flash-Next GSQ-RCO IQ3_S（ISTA-DASLab），Strata 的 native pack，24,576 個 routed expert（48 層 × 512）
- **引擎**：Strata 0.1.43.1，fork 分支 `gfx906-ep`。prefill 端是 CUDA sm_86 build，decode 端是 HIP gfx906 build，兩者同一份原始碼。
- **decode 端設定**：B 配置。MI50 #0 放 dense 層、KV、MTP，MI50 #1 放其餘 expert（`--expert-cache-device1 auto --remote-expert-opt`）。`--spec 4 --spec-min-p 0.5 --mtp`、`--kv int8`、`--max-context 262144`、`--kv-resident 262144`。環境變數：`STRATA_GFX906_TOPK_SPLIT=1 STRATA_GR_SPLIT=1 STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1`
- **GPU 政策**：MI50 sclk 上限 1,339 MHz、功耗上限 170 W（為了散熱和供電穩定，不是效能最佳值）。3070 鎖頻在 1,500–2,100 MHz、記憶體 7,001 MHz。
- **decode**：greedy（temperature 0），關閉 thinking，除非另外註明

## TTFT 與總時間：32K prompt

| | prefill | session 交接 | TTFT | decode 200 token | 總時間 |
|---|---|---|---|---|---|
| 只用 2× MI50（B 配置） | 80.2 s（400 tok/s） | — | 約 80 s | 4.8 s | 85.2 s |
| PD，session 檔放 SATA SSD | 21.2 s（1,525 tok/s） | 存 2.31 s + 讀 2.50 s | 約 26 s | 4.3 s | 30.4 s |
| **PD，session 檔放 `/dev/shm`** | 21.2 s | 存 1.23 s + 讀 0.94 s | **約 24 s** | 4.4 s | **27.8 s** |

TTFT 是 prefill 加上 session 交接，再加上 decode 端讀入 session 後補讀的 7 個 token（約 0.3 s）。

32K 的 session 檔是 725,748,212 bytes。MI50 讀進來之後，32,057 個 token 裡重用了 32,050 個，自己只補讀 7 個。

## prefill 速度隨深度（3070，CUDA build）

| prompt | 4K | 16K | 24K | 32K | 40K | 48K | 64K | 96K | 128K | 192K | 258K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tok/s | 1,053 | 1,520 | 1,558 | 1,600 | 1,603 | 1,600 | 1,598 | 1,581 | 1,549 | 1,496 | 1,439 |

（第一輪，`--kv-resident 32768`。KV 全放 VRAM 之後重量：32K 1,645、128K 1,651、258K 1,548。）

對照：只用兩張 MI50 prefill，A 配置（分層）32K 是 622 tok/s，B 配置是 396 tok/s。

## decode 隨深度（MI50，300 token 回答，PD）

| | 短 | 32K | 64K | 128K | 192K | 258K |
|---|---|---|---|---|---|---|
| 原始設定（`--kv-resident 32768`） | 39.0 | 42.5 | — | 34.7 | — | 30.1 |
| KV 全放 VRAM | 40.7 | 44.7 | — | 39.6 | — | 34.4 |
| **KV 全放 VRAM + 兩段式 top-k** | 39.2 | 46.5 | 40.0 | 40.5 | 39.3 | 39.4 |

題目是用 tokenizer 從一份 28.2 萬 token 的語料（文件加原始碼，沒有重複內容）精確切出各深度，再加上「請詳細說明上文…至少 400 字」。表中的「短」是一題介紹台灣夜市的題目，MTP 在這題的命中率偏低（110/198），所以短 prompt 的數字比下一節的一般對話低。

## decode 依任務類型（MI50，PD 的 decode 引擎直接請求）

每一類 5 題，每題最多 400 token，greedy：

| 類型 | 每個 window 產出的 token | decode tok/s |
|---|---|---|
| 一般對話 | 2.13 | 47.3 |
| 程式 | 3.26 | **60.6** |
| 數學 | 3.30 | **60.6** |

每個 window 產出 = decode token 數 ÷（decode token 數 − 草稿接受數）。程式和數學的內容可預測性高，MTP 猜中的比例也高。

程式和數學題就是 `scripts/bench_tasks.py` 裡的題目。一般對話的 47.3 是用我們內部的另一組 5 題量的；腳本裡的一般對話題目是另外寫的，用 Docker 映像版的引擎（`docker/`）跑整支腳本：一般對話 43.4（每個 window 1.93）、程式 61.7、數學 61.1 tok/s。

## 與其他硬體的單請求對照

只列同一個引擎（Strata）、同一個量化（Flash-Next IQ3_S）、單一請求的實測。

| 硬體 | 32K prefill（tok/s） | decode，短～32K（tok/s） | decode，128K（tok/s） | 來源 |
|---|---|---|---|---|
| RTX 5090 | 4,168 | 142.2 | 127.2 | [社群，2026-10-02](https://github.com/Niko1221/Strata/tree/main/bench/results/2026-10-02-community-rtx-5090-iq3s) |
| 2× TITAN RTX | 1,642 | 80.2 | 70.6 | [社群，2026-10-08](https://github.com/Niko1221/Strata/tree/main/bench/results/2026-10-08-community-2x-titan-rtx-0.1.40.3) |
| R9700 32 GB | 750 | 95.6 | 87.6 | [社群，2026-10-08](https://github.com/Niko1221/Strata/tree/main/bench/results/2026-10-08-community-r9700-linux) |
| **PDHyperMix（2× MI50 + 3070）** | **1,597–1,645** | **46.5（32K）；一般對話 47.3／程式 60.6** | **40.5** | 本頁 |
| Strix Halo（桌機） | 約 1,250（8K） | 59.7（8K） | 42.8 | [Strata 文件 STRIX_HALO.md](https://github.com/Niko1221/Strata/blob/main/docs/STRIX_HALO.md) |
| RX 9070 XT 16 GB | 601 | 45–46 | — | [社群，2026-10-02](https://github.com/Niko1221/Strata/tree/main/bench/results/2026-10-02-community-rx9070xt-windows)（引擎 0.1.35） |
| 只用 2× MI50 | 396 | 約 48 | 34.7 | 本頁 |

單請求時，PDHyperMix 的 prefill 接近 2× TITAN RTX，decode 接近 Strix Halo、RX 9070 XT 這一級。

這是粗略對照：

- 各份報告的引擎版本（0.1.35–0.1.43）、生成長度（256 或 300 token）和題目都不同。
- decode 受 MTP 命中率影響很大，而命中率取決於題目；我們自己的數字就落在 39–60 tok/s 之間。
- Strix Halo 那一列取自 Strata 文件的基準值，不是社群報告；8K prefill 不等於 32K prefill。
- 量化不同的報告（Strix Halo 筆電的 UD-IQ4_XS、RX 9070 的 IQ3_XXS、V100 的 Q2_0 等、單張 MI50 的 IQ2_XS）沒有放進來。

## 兩段式 top-k（gfx906 fork）

`qsa_select_bench`，3 個 query，容量 262,144，單次呼叫時間：

| 上下文 | 原本（ref） | wide kernel | **兩段式** |
|---|---|---|---|
| 258K | 0.743 ms | 0.324 ms | **0.119 ms** |
| 131K | 0.533 ms | 0.211 ms | **0.086 ms** |
| 64K | 0.309 ms | 0.114 ms | **0.082 ms** |
| 40K | 0.211 ms | 0.092 ms | **0.076 ms** |

挑出的 id 在所有長度都跟 ref 完全一樣。放進引擎實測：258K 的 `scores+topk` 從每個 window 12.24 ms 降到 4.28 ms，decode 從 33.7 提升到 39.4 tok/s，固定 prompt 的 greedy 輸出跟 ref build 逐字一致。

## 交接與 RAM

- **session 檔大小**：32K 約 0.73 GB、64K 1.24 GB、128K 2.24 GB、258K 4.18 GB
- **存檔加讀檔**（`/dev/shm`，兩輪的範圍）：32K 2.1–2.3 s、128K 5.3–5.5 s、258K 10.0–11.6 s
- **SATA SSD 冷讀**：533 MB/s。session 檔用 `O_DIRECT` 寫入，page cache 裡是 0 頁。
- **RAM**（兩個引擎在線時）：used 7 GB、shared 50 GB（共用 expert 區）、available 約 34 GB

## 兩個請求同時進來

兩個不同的 32K prompt，各 decode 300 token，經路由同時或依序送出：

| | 合計 | A：TTFT / 完成 | B：TTFT / 完成 |
|---|---|---|---|
| 依序 | 62.2 s | 21.2 / 28.7 s | 26.0 / 33.5 s |
| 同時 | 52.3 s | 21.7 / 41.7 s | 44.7 / 52.1 s |

同時送時，A 的 decode 從 7.4 s 拖到 20.3 s，這段時間 3070 正在讀 B 的 prompt。推估原因是 host 記憶體或 PCIe 的爭用，還沒驗證。

## prefill 卡

RTX 3070 是改裝的 16 GB 版。它也用 [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) 解鎖了 FP16/BF16 的指令發射率：cuBLAS 8192³ GEMM 測得 FP16 80.4、BF16 81.8、TF32 44.3 TFLOPS。

- 這張卡解鎖前沒有量過，所以**解鎖對 prefill 的貢獻沒有拆開量**。
- 依原始碼推估：dense 層的 GEMM 和 QSA attention 走 FP16/BF16，會受益；MoE expert 走 INT8，不受影響。
- 不解鎖的 3070，或其他 prefill 卡，PD 一樣能運作，只是 prefill 的數字會不同。

## 試過、沒有採用的方向

| 方向 | 結果 |
|---|---|
| 不改輸出的融合開關（`LFUSE`、`HEAD_MIX_MULTI`、`VERIFY_QDEDUP`、`SH_STREAM`、`FUSE_CONVAB`） | window 時間差在 0.5 ms 以內 |
| `--spec` 3／5／6、`--spec-min-p` 0.3／0.7 | 換算後 46–50 tok/s，差在 3% 以內 |
| 近似路由（`STRATA_ROUTE_RESIDENT`、`STRATA_ROUTE_PRIOR`） | 更慢（43.9–46.2 對 48.7），輸出相似度只有 0.19–0.32 |
| dataflow 側流（`STRATA_DF_BRANCH`） | 44.65，變慢 |
| DFlash2 草稿模型（llama.cpp，抓取方式已修正） | 每個 window 產出：猜 3 個時 2.18／3.23／3.26（一般／程式／數學），MTP 是 2.53／3.52／3.52；猜 7 個時程式題 4.99，但驗證 8 個 token 的成本推估會抵銷掉 |
| 在 MI50 上做分層 prefill | 32K 從 403 提升到 625 tok/s，但遠低於 3070 的 1,600 |
