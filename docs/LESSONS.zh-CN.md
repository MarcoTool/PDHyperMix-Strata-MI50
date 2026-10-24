# 限制、事故与排查

按“最可能让你卡住的程度”排列。

## 1. RAM 是硬上限，任何额外的 Strata 进程都要绑定共享 expert 区

PD 在线时，`/dev/shm` 里有一份约 47 GiB 的共享 expert 区。这时如果另外跑 `strata generate`、benchmark 或 profiling，**一定要加上同一个 `--shared-expert-arena`**，或者先停掉两个引擎、删掉共享 expert 区再跑。

**我们发生过一次：** 跑剖析时漏了这个参数，新的进程又配了一份 47 GB。

1. 内存超量，systemd-oomd 连杀两个容器。
2. amdgpu 的 `svm_range_restore_work` 和 `amdkfd_restore_userptr_worker` 卡进 D 状态，持续 368 秒。
3. D 状态检测服务停止喂 softdog，`kernel.panic=60` 把整台机器重启。

**动手前先算总量**：共享 expert 区 47 GiB（tmpfs 算在 RAM 里）＋每个引擎 5–10 GB ＋其他服务。启动脚本在 RAM 不够时会拒绝执行。

## 2. `/dev/shm` 的默认大小放不下共享 expert 区加 session 文件

tmpfs 的默认上限是物理 RAM 的一半，这台是 48 GiB。共享 expert 区 46.84 GiB 加上 session 文件（32K 约 0.7 GiB，Strata 还要求保留 4 GiB），会超过上限，保存就会失败（HTTP 507）。

```bash
sudo mount -o remount,size=64G /dev/shm   # 重启后会恢复原本的大小
```

这只是“允许用到多少 RAM”的上限，不会预先占用。

## 3. session 指纹：两个引擎的配置必须完全一致

session 文件里有一个配置指纹（`conversation_file.cpp` 的 `session_config_fingerprint`），内容包含引擎版本、后端、`--kv`、`--max-context`、`--kv-resident`、MTP window、RoPE 参数和 control vector。

只要有一项不同，读文件就会失败：“saved with another engine configuration”。例如我们只改了 decode 端的 `--kv-resident`，整轮实测就全部作废。两个引擎的配置请由同一个 `pd_cfg.py` 生成。

CUDA 版和 HIP gfx906 版之间可以互相交接：gfx906 build 是把 CUDA 源码经由 HIP 兼容层编译的，两边报告的后端相同，expert 区的 pack hash 也一致。

## 4. session 文件不要放 SSD

Strata 存 session 文件时用 `O_DIRECT` 加 `fsync`，完全不经过 page cache。放在 SATA SSD 上，32K 存盘加读盘要 4.8 秒；放在 tmpfs 只要 2.2 秒。剩下的时间大多花在 GPU 状态搬回 host 和 hash 计算，要再压就得改 Strata 的程序代码。

## 5. 两个请求同时进来只快 16%

引擎一次只能处理一个请求，因为 slot 存档和并行请求模式（`--batch`）互斥。路由排成两段流水线，但 3070 prefill 的同时，MI50 的 decode 会被拖慢约 2.7 倍。估算原因是 host 内存或 PCIe 的争用，还没验证。

## 6. PD 的输出不保证逐字可重现

同一个 prompt 在不同时间送进来，回答的措辞可能略有不同。原因估算有两个：

- 3070（CUDA）和 MI50（HIP）的浮点舍入不同。
- 3070 在 prefill 之前处理过什么，会影响它的 expert 缓存状态，连带影响运算路径。

我们验证过：直接请求 MI50 引擎时，输出可以逐字重现；同一个 3070 状态下重复交接，也可以逐字重现。

## 7. 魔改 3070：开机时撞到 Xid 119，连关机都会卡住

这张魔改的 16 GB 3070 有已知问题：在长时间加载的过程中掉进低功耗状态时，会出现 Xid 119／154（GSP RPC 超时）。所以我们用一支开机服务先锁频。

有一次热重启时，Xorg 比锁频服务早碰到 3070，GPU 进入“requires reset”状态。之后关机时，`gdbus` 和 `plymouthd` 都卡在这张卡上（D 状态），只能断电重启。

**建议**：headless 主机不要让显示相关程序（Xorg、gdm、plymouth）碰 prefill 卡，或者让锁频服务排在它们之前启动。

## 8. rocprofv3 剖析不了 Strata 的 decode

decode 里有 device 端的 doorbell，以及会等待标志的常驻 kernel。rocprofv3 的 queue 拦截会一直等它们的完成信号，永远等不到（“Async signal handler still waiting on signal”）。

替代方案是 Strata 内建的工具：

- `STRATA_DECODE_TIMING=1`：每个 window 的时间拆解
- `STRATA_VERIFY_PROFILE=1`：各阶段的 GPU 时间
- `STRATA_VERIFY_NODES=1`：graph 的节点组成
- `STRATA_DUMMY_NODES=k`：量每个 graph 节点的成本

## 9. 测试要看题目类型

MTP 在程序和数学题上，每个 window 产出 3.3 个 token；在一般对话上只有 2.1 个。只用一种题目测试，会把整体速度估算错 25% 以上。

另外，用 4 段平均值比较配置时噪声很大：同一个配置重启后，其中一段可能从 63 掉到 48。比较配置时改看 `STRATA_DECODE_TIMING` 的每个 window 耗时。
