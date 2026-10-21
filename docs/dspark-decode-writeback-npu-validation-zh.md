# DSpark 场景：Decode KV 回写与下一轮 Prefill 复用的 NPU 验证背景

整理日期：2026-10-21。

用途：交给其他模型或工程师，作为 NPU 环境定位和验收的背景。本文是源码分析和测试建议，
不是已完成 NPU 验证的支持声明。DSpark 为正确名称，历史对话中的 DSpare 指同一对象。

## 1. 任务目标与必须区分的两个问题

目标：第一轮 Decode 开启 DSpark，将已提交且完成目标模型计算的回答 KV 写入共享池；
第二轮 Prefill 输入包含第一轮实际历史，从池恢复目标 KV，减少历史重算和 TTFT。

必须分开验收：

1. 目标模型 KV 的 Decode 回写和下一轮 Prefill 复用。
2. 下一轮 Decode 若继续开启 DSpark，其 draft 历史上下文能否正确初始化和继续使用。

第 1 项不在算法上要求 Prefill 加载 drafter，也不要求把 draft KV 一起保存。
第 2 项可能需要恢复或重建 draft 上下文，不能因为第 1 项成功就认为第 2 项成立。

## 2. 两类 KV：相同 token 位置不意味着相同数据

| 缓存 | 产生方式 | 使用者 |
|---|---|---|
| 目标模型 KV | 目标模型对历史进行计算，包括对候选 token 进行验证 | 目标模型 attention、下一轮 Prefill |
| DSpark context/draft KV | 使用目标模型辅助 hidden states，经 drafter 自己的投影产生 | drafter 的候选生成 |

DSpark 提出候选后，目标模型验证会生成目标模型自己的 KV。不能把 drafter 的 KV 当作目标 KV。
只允许发布已提交、已完成有效计算的历史；被拒绝的候选位置和未实际计算的采样 token 不应进入有效保存范围。

开启 DSpark 不会使有效目标 KV 天然失去复用资格。复用仍要求目标权重、历史 token、位置语义、
KV dtype/量化、布局等兼容。不同批形状、TP 和数值计算路径不保证逐字节或逐 token 完全相同。

“加载 drafter”不等于“在 Prefill 执行投机生成”。某些 PD 协议在 P 加载 drafter，仅用于构建
draft context KV 并传给 D；这是实现选择，不是通用的 Prefill 要求。

## 3. 已检查的源码快照与本地位置

分析期间 main 发生推进，结论固定到下列实际检查快照，不自动代表未来 main 或任意 release。

- vLLM-Ascend：`c361fb4990aa9c32a9609e34595d787bb4144f2a`
  - 本地只读分析副本：`D:\code\Model inference\tmp\dspark-ascend-audit`
- vLLM：`0495a3de3226850d1c2c6c553a52c6e10830d14d`
  - 本地只读分析副本：`D:\code\Model inference\tmp\dspark-vllm-audit`
- 本次 UCM 开发快照：`631655fd05cb7118d8127b43881806a86b2496e3`
  - 本地：`D:\code\Model inference\ucm-decode-writeback`
  - 社区 PR：<https://github.com/ModelEngine-Group/unified-cache-management/pull/1542>

这些 SHA 用于定位源码，不表示该 vLLM/Ascend 组合已通过安装兼容性或 NPU 验收。
实际测试必须记录安装包版本、Git SHA、import 路径及本地补丁；不要把 0.23/0.26 行为套用到主线。

## 4. AscendStore 普通非 Layerwise 路径的源码发现

### 4.1 DSpark context KV 有实际 cache tensor 承载

文件：`vllm_ascend/models/deepseek_v4/dspark.py`。

调用关系：

```text
target aux hidden states
  → combine_hidden_states()
  → precompute_and_store_context_kv()
  → _store_standard_swa_kv()
  → draft 层 swa_cache_layer.kv_cache
```

DeepSeek-V4 DSpark 权重位于 `mtp.*` 名称空间，但执行的是 block drafter，不等于普通串行 MTP。
若历史 context KV 已正确恢复，不必仅为了同一历史再次产生整段辅助 hidden states。

### 4.2 普通 Store 可能保存 target 与 draft 数据

文件：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`。

- `register_kv_caches()` 注册收到的缓存 tensor。
- `_infer_cache_group_metadata()` 按组和层构建地址、长度、步长以及组件顺序。
- 普通非 Layerwise 路径没有按 `mtp.*` 统一排除 draft 层。
- `metadata.py::infer_cacheable_group_ids()` 根据组的前缀可缓存性选择保存/恢复范围。
- DeepSeek-V4 draft SWA 使用可参与前缀缓存的规格；仍需检查部署实际生成的组和 flags。

结论边界：在实际 cache plan 包含 draft tensor、缓存组参与传输且读写布局兼容时，
普通组对象可以包含 draft context KV。不能仅因未走 DSparkPrefixCache 专用入口就判定不支持。

### 4.3 Scheduler 推进与保存时序

Scheduler 文件：`.../ascend_store/pool_scheduler.py`。

```text
build_connector_meta()
  → _process_running_cached_request()
  → 按 num_computed_tokens 与调度范围更新 tracker
  → 非 Layerwise 限制到 all_token_ids 已提交范围
  → _build_req_meta() / ReqMeta.from_request_tracker()
```

开启 `save_decode_cache` 后，无新 block 的步骤仍处理；有效完整 hash 单元才可形成保存范围。
Scheduler 计划日志不是保存成功，tracker 范围也不等于设备数据已经完成传输。

MRV2 的关键顺序：

```text
target forward → sample/rejection → speculator.propose()
  → 更新 context KV / draft forward
  → kv_connector.finalize_step() → wait_for_save()
```

- vLLM：`vllm/v1/worker/gpu/model_runner.py::sample_tokens()` 延迟 connector finalization。
- vLLM：`vllm/v1/worker/gpu/kv_connector.py::finalize_step()` 调用保存。
- Ascend MRV1：`vllm_ascend/worker/v1/model_runner.py` 也在 draft 后调用 `finalize_kv_connector()`。
- Store Worker 在 `wait_for_save()` 记录 NPU event；发送线程 `_handle_stored_request()`
  等待 event 后读取源地址并调用后端 PUT。

这是源码时序依据，仍需验证部署实际 runner、graph、stream 和版本确实走到该路径。

### 4.4 下一轮恢复

```text
get_num_new_matched_tokens()
  → LookupKeyClient / Worker lookup_scheduler()
  → coordinator 检查各必要组的可达命中
  → LoadSpec + 本地 block 分配
  → KVPoolWorker.start_load_kv()
  → 按组构造目标地址 → backend.get()
  → 未命中后缀 target forward → 更新新后缀 context KV → 后续 draft/verify
```

不能只看最长单组命中；必要目标组及实际需要的 draft/window 状态必须与恢复边界一致。
尾部为 draft 安全重计算保留部分 token 并不等于复用失效。具体裁剪以运行版本为准。

## 5. 一端有 drafter、另一端没有：真正的风险是对象兼容性

在多轮复用中，“写入端”通常是第一轮 Decode，“读取端”是第二轮 Prefill。

- 若 target 与 draft 对象分离，P 只读取兼容的 target 对象，不需要 drafter。
- 若一个组对象把 target 与 draft 的多个层组件打包，D/P 层集合不同，可能导致 key、组编号、
  对象大小或组件顺序不兼容。不能仅凭同一个 token hash 就直接加载。
- 必须同时打印写入和读取侧的 group/layer/component 描述，确认真正可互读。
- draft 模型不同，不必影响纯 target 对象；但不能复用不兼容的 draft 对象。

因此不要使用“一端没 drafter，所以目标 KV 必然不能复用”这一推断。

## 6. SfaRemoteD2HConnector PD 路径的额外限制

这不是所有 PD Connector 的通用限制。

- `uses_sfa_dspark_kv_transfer()` 识别 DSpark + SfaRemoteD2HConnector（含 MultiConnector）。
- 纯 P producer 若没有找到配对 target/draft 前缀 provider，AscendMultiConnector 保持
  `_requires_full_dspark_prompt=True`，`get_num_new_matched_tokens()` 直接返回 `(0, False)`。
- 配对 `configure_dspark_prefix_cache()` 要求 producer + Layerwise GVA；专用对象使用
  `dspark-prefix-v1@...` key，范围限制在 prompt。
- 不能推断普通非 Layerwise D 写入的对象，可直接供配对 Layerwise P 使用。
- 该路径可能要求 P 加载 drafter、产生 prompt context KV 后再传 D；这与 P 自己持续生成回答无关。

所以出现“D PUT 成功但 P 没有外部命中”时，先检查 MultiConnector 是否主动禁止了 lookup，
再检查 key；不要一开始就认定 token hash 不一致或通信失败。

## 7. 对本次 UCM PR 的正确解释

本次 `replicated_mla_layout.py::target_layers()` 排除识别出的 draft/MTP 层，主要定义 target 记录。

- 排除 draft 层并不天然妨碍“DSpark 下保存目标 KV，下一轮无 drafter 的 P 复用目标 KV”。
- 尚未验证 DSpark 下目标 KV 生成/回滚/保存时序，以及下一轮 D 的 context 初始化全链路，
  因而暂不承诺完整 DSpark PD 支持。
- 不应把“无真实 DSpark 适配与验证”写成“算法上必须保存所有 draft KV”。
- 当前开关开启时，各角色均保留实际 Prefill/re-prefill 保存；不要再沿用 consumer 跳过 Prefill 的旧说明。
- UCM 的 169 项 CPU 测试、22 项子用例通过记录不等于真实 DSpark/模型/NPU 验证。

## 8. NPU 测试前必须收集的信息

1. 目标 checkpoint、revision、量化和 KV dtype；draft checkpoint、辅助层选择、投机宽度。
2. vLLM/Ascend/UCM SHA、安装路径、runner 版本、async scheduling、graph/eager 配置。
3. P/D 的 DP/TP/PP/PCP/DCP、实际 Connector 类与 MultiConnector 子项顺序及各自 kv_role。
4. Store 后端、P/D 是否使用同一个池、Layerwise 开关、key namespace 与记录布局。
5. 每组 group_id、层名、target/draft 归属、block size、存储步长、组件字节数、cacheability。
6. 实际 hash/transfer 粒度、读取/保存阈值。不要固定假设 128 或 16384。

先以同构 TP、低并发、固定输入验证，再扩展异构 TP、并发、graph 和动态验证长度。

## 9. 最小功能验收：分开证明写入、读取、正确性、收益

### 9.1 第一轮写入

- D 开 DSpark，生成足够长的真实回答，跨过多个实际保存边界。
- 记录原始 prompt 长度 P、committed 长度、实际已计算范围和 backend PUT 完成结果。
- 至少一个成功完整单元必须覆盖 `[P, ... )` 的回答范围，而不只是回存 prompt。
- 最后采样 token 可能尚未生成自身 KV；不足完整单元的尾部不要求强制保存。

### 9.2 第二轮读取

- 回传第一轮真实输出及必要 reasoning 字段，不使用 `ignore_eos=true` 强行拉长 Chat 输出。
- 检查渲染后 token 前缀，排除 EOS、思考标签、换行与历史裁剪导致的 key 差异。
- 在独立测试服务清理/隔离 P 的本地前缀缓存，保留共享池；避免本地 draft 状态掩盖缺失。
- 确认 P 的成功 GET 范围进入第一轮回答，且实际 Prefill 计算量减少。
- 排除 P 自己重新计算并保存同一对象造成的假阳性；通过请求时间、角色、job/key 对照确认来源。
- 不要清理生产共享池或重启生产实例；使用独立测试实例/命名空间。

### 9.3 正确性与 DSpark 初始化

- 对同一份第二轮输入，在同一读取端 TP 比较冷缓存重算与外部加载两种模式。
- 保持两组 DSpark 配置一致；先区分一般数值差异与缓存错误，必要时比较 logits/概率。
- 检查下一轮首次及后续 draft/verify 的 context 有效长度、已恢复范围、接受/拒绝行为。
- 必须说明历史 context 从池恢复、经 P 传输，还是重算；不能读取未初始化历史。
- 检查是否静默退回整段重算：输出正确但不减少计算，不算达成功能目标。

### 9.4 性能

- 保存开关关闭/开启使用隔离的冷池状态，并重放相同历史，避免前一次实验污染基线。
- 成对报告第二轮及以后 TTFT、实际外部加载量、Prefill 计算量。
- 同时报告第一轮 Decode TPOT、写流量、请求总时延；保存同步开销可能抵消部分收益。
- 最终性能测试关闭高频 DEBUG；不要把日志开销计成特性固有开销。

## 10. 推荐测试矩阵

| 场景 | 主要确认点 |
|---|---|
| D 不开 DSpark → P target-only | 基础目标 KV 回写/读取基线 |
| D 开 DSpark → P target-only | target 记录是否摆脱 draft 层集合依赖 |
| D 开 DSpark → P 复用 → 下一轮 D 继续 DSpark | context 恢复/重建链路与正确性 |
| 接受 0/部分/全部候选 | 拒绝位置不发布为有效历史，已接受目标 KV 可复用 |
| hash 边界前后、短输出、长输出 | 粒度、窗口保留与尾部重算行为 |
| 抢占恢复与请求结束 | 源 block 生命周期、异步 PUT 完成和旧回执隔离 |
| SFA PD 特定组合 | 是否先被 `_requires_full_dspark_prompt` 拦截 |

## 11. 建议日志字段及判读顺序

日志字段：request_id、端点角色、阶段、job_id/代次、目标/draft 标识、group_id、layer_names、
token_range、computed/committed/scheduled 范围、hash/key、对象字节数、PUT/GET 真实完成状态、
lookup 跳过原因、恢复边界、下一次 draft context 有效长度。

判读顺序：

```text
是否允许查询 → key/格式是否兼容 → 必要组是否完整 → GET 是否成功
  → target 是否减少重算 → draft 历史是否有效 → 输出正确性 → TTFT/TPOT 收益
```

计划日志、对象存在、单组命中、API usage 或 TTFT 单独均不是完整验收证据。

## 12. 上游已有测试与证据边界

AscendStore 的 `tests/ut/distributed/ascend_store/test_request_lifecycle.py` 包含 `dspark` 参数、
模拟 draft 层、保存开关、拒绝回滚、清理本地缓存和读取 Decode 完成页面的测试。
其模型计算使用确定性 CPU 字节写入，Store 为模拟后端；不能替代实际 DSpark 投影/attention/NPU 验证。
本次分析没有执行该上游测试，不能报告它当前通过。

## 13. 固定版本源码入口

- [DeepSeek-V4 DSpark 模型](https://github.com/vllm-project/vllm-ascend/blob/c361fb4990aa9c32a9609e34595d787bb4144f2a/vllm_ascend/models/deepseek_v4/dspark.py)
- [Store Worker 注册/PUT/GET](https://github.com/vllm-project/vllm-ascend/blob/c361fb4990aa9c32a9609e34595d787bb4144f2a/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py)
- [Store Scheduler](https://github.com/vllm-project/vllm-ascend/blob/c361fb4990aa9c32a9609e34595d787bb4144f2a/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py)
- [Store 元数据](https://github.com/vllm-project/vllm-ascend/blob/c361fb4990aa9c32a9609e34595d787bb4144f2a/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py)
- [MRV2 draft 后 finalize](https://github.com/vllm-project/vllm/blob/0495a3de3226850d1c2c6c553a52c6e10830d14d/vllm/v1/worker/gpu/model_runner.py)
- [Ascend MRV1 runner](https://github.com/vllm-project/vllm-ascend/blob/c361fb4990aa9c32a9609e34595d787bb4144f2a/vllm_ascend/worker/v1/model_runner.py)
- [SFA PD MultiConnector 查询限制](https://github.com/vllm-project/vllm-ascend/blob/c361fb4990aa9c32a9609e34595d787bb4144f2a/vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py)
- [P 侧 draft context 构建/传输](https://github.com/vllm-project/vllm-ascend/blob/c361fb4990aa9c32a9609e34595d787bb4144f2a/vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/dspark_context.py)
- [DSpark 专用前缀对象](https://github.com/vllm-project/vllm-ascend/blob/c361fb4990aa9c32a9609e34595d787bb4144f2a/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/dspark_prefix_cache.py)
- [CPU 生命周期测试](https://github.com/vllm-project/vllm-ascend/blob/c361fb4990aa9c32a9609e34595d787bb4144f2a/tests/ut/distributed/ascend_store/test_request_lifecycle.py)

## 14. 给后续模型的一句话结论

目标 KV 回写供下一轮 Prefill 复用，在原理上不要求 Prefill 加载 drafter；DSpark 的主要附加风险是
投机有效范围与保存时序、Store 对 target/draft 的布局耦合，以及下一轮 D 的 draft 历史初始化。
普通非 Layerwise AscendStore 有相应实现基础，但指定部署仍需 NPU E2E；SFA PD 等特定路径可能
明确禁止外部命中，必须按实际 Connector 分析。不要把“未验证”改写成“已支持”或“必然不支持”。
