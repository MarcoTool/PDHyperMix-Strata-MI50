# 限制、事故與排解

按「最可能讓你卡住的程度」排列。

## 1. RAM 是硬上限，任何額外的 Strata 行程都要綁共用 expert 區

PD 在線時，`/dev/shm` 裡有一份約 47 GiB 的共用 expert 區。這時如果另外跑 `strata generate`、benchmark 或 profiling，**一定要加上同一個 `--shared-expert-arena`**，或者先停掉兩個引擎、刪掉共用 expert 區再跑。

**我們發生過一次：** 跑剖析時漏了這個參數，新的行程又配了一份 47 GB。

1. 記憶體超量，systemd-oomd 連殺兩個容器。
2. amdgpu 的 `svm_range_restore_work` 和 `amdkfd_restore_userptr_worker` 卡進 D 狀態，持續 368 秒。
3. D 狀態偵測服務停止餵 softdog，`kernel.panic=60` 把整台機器重開。

**動手前先算總量**：共用 expert 區 47 GiB（tmpfs 算在 RAM 裡）＋每個引擎 5–10 GB ＋其他服務。啟動腳本在 RAM 不夠時會拒絕執行。

## 2. `/dev/shm` 的預設大小放不下共用 expert 區加 session 檔

tmpfs 的預設上限是實體 RAM 的一半，這台是 48 GiB。共用 expert 區 46.84 GiB 加上 session 檔（32K 約 0.7 GiB，Strata 還要求保留 4 GiB），會超過上限，存檔就會失敗（HTTP 507）。

```bash
sudo mount -o remount,size=64G /dev/shm   # 重開機後會恢復原本的大小
```

這只是「允許用到多少 RAM」的上限，不會預先佔用。

## 3. session 指紋：兩個引擎的設定必須一模一樣

session 檔裡有一個設定指紋（`conversation_file.cpp` 的 `session_config_fingerprint`），內容包含引擎版本、後端、`--kv`、`--max-context`、`--kv-resident`、MTP window、RoPE 參數和 control vector。

只要有一項不同，讀檔就會失敗：「saved with another engine configuration」。例如我們只改了 decode 端的 `--kv-resident`，整輪量測就全部作廢。兩個引擎的設定請由同一個 `pd_cfg.py` 產生。

CUDA 版和 HIP gfx906 版之間可以互相交接：gfx906 build 是把 CUDA 原始碼經由 HIP 相容層編譯的，兩邊回報的後端相同，expert 區的 pack hash 也一致。

## 4. session 檔不要放 SSD

Strata 存 session 檔時用 `O_DIRECT` 加 `fsync`，完全不經過 page cache。放在 SATA SSD 上，32K 存檔加讀檔要 4.8 秒；放在 tmpfs 只要 2.2 秒。剩下的時間大多花在 GPU 狀態搬回 host 和 hash 計算，要再壓就得改 Strata 的程式碼。

## 5. 兩個請求同時進來只快 16%

引擎一次只能處理一個請求，因為 slot 存檔和平行請求模式（`--batch`）互斥。路由排成兩段流水線，但 3070 prefill 的同時，MI50 的 decode 會被拖慢約 2.7 倍。推估原因是 host 記憶體或 PCIe 的爭用，還沒驗證。

## 6. PD 的輸出不保證逐字可重現

同一個 prompt 在不同時間送進來，回答的措辭可能略有不同。原因推估有兩個：

- 3070（CUDA）和 MI50（HIP）的浮點捨入不同。
- 3070 在 prefill 之前處理過什麼，會影響它的 expert 快取狀態，連帶影響運算路徑。

我們驗證過：直接請求 MI50 引擎時，輸出可以逐字重現；同一個 3070 狀態下重複交接，也可以逐字重現。

## 7. 改裝 3070：開機時撞到 Xid 119，連關機都會卡住

這張改裝的 16 GB 3070 有已知問題：在長時間載入的過程中掉進低功耗狀態時，會出現 Xid 119／154（GSP RPC 逾時）。所以我們用一支開機服務先鎖頻。

有一次熱重開時，Xorg 比鎖頻服務早碰到 3070，GPU 進入「requires reset」狀態。之後關機時，`gdbus` 和 `plymouthd` 都卡在這張卡上（D 狀態），只能斷電重開。

**建議**：headless 主機不要讓顯示相關程式（Xorg、gdm、plymouth）碰 prefill 卡，或者讓鎖頻服務排在它們之前啟動。

## 8. rocprofv3 剖析不了 Strata 的 decode

decode 裡有 device 端的 doorbell，以及會等待旗標的常駐 kernel。rocprofv3 的 queue 攔截會一直等它們的完成訊號，永遠等不到（「Async signal handler still waiting on signal」）。

替代方案是 Strata 內建的工具：

- `STRATA_DECODE_TIMING=1`：每個 window 的時間拆解
- `STRATA_VERIFY_PROFILE=1`：各階段的 GPU 時間
- `STRATA_VERIFY_NODES=1`：graph 的節點組成
- `STRATA_DUMMY_NODES=k`：量每個 graph 節點的成本

## 9. 量測要看題目類型

MTP 在程式和數學題上，每個 window 產出 3.3 個 token；在一般對話上只有 2.1 個。只用一種題目量，會把整體速度估錯 25% 以上。

另外，用 4 段平均值比較設定時雜訊很大：同一個設定重啟後，其中一段可能從 63 掉到 48。比較設定時改看 `STRATA_DECODE_TIMING` 的每個 window 耗時。
