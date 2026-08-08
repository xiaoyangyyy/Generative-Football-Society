# GFS 复杂足球系统：统一产品架构 V2

## 1. 产品定义

GFS 不是几个相互独立的实验脚本，而是一个以比赛仿真为核心、由证据控制可选智能层的复杂足球系统。用户入口只有 **GFS Studio**；训练、校准、封闭评测和 LLM 试验属于侧向研发流水线，不能直接获得产品运行权。

```text
                         GFS Studio（统一控制面）
                    会话 / 比赛 / 任务 / 报告 / 证据
                                  |
               +------------------+------------------+
               |                                     |
        比赛领域运行面                         研发与证据面
  规则、物理、心理、战术、赛制          数据集 -> 训练任务 -> 候选件
               |                              -> 封闭评测 -> 决策件
       +-------+-------+                             |
       |               |                             |
  世界模型适配器   认知/LLM 适配器 <----------------+
  默认关闭、证据授权  严格模式必须真实调用成功
```

依赖方向是单向的：Studio 编排领域层；领域层不反向依赖 Studio。训练只生成候选件；封闭证据生成决策件；只有被接受的决策件才能启用适配器。稳定模式永远不依赖研究模块。

## 2. 五个层次

1. **产品控制面**：持久化 Studio 会话，统一显示模式准备度、训练任务、模型决策、比赛报告和认知审计。
2. **比赛领域层**：球队、球员、球、物理、裁判、战术、心理、微观比赛和锦标赛，是稳定产品的主体。
3. **智能适配层**：世界模型给出建议；独立射门头只消费冻结表征；认知层在显著事件上生成教练计划。三者都不是比赛真相源。
4. **研发流水线**：数据清单、训练、断点、日志、候选产物。长任务可观察、可恢复、可协作停止。
5. **证据与发布层**：dev 只用于开发，sealed test 只用于一次性裁决。发布清单是生产权威，评测报告不能自行改变稳定产品。

## 3. 三种产品模式

| 模式 | 稳定比赛引擎 | 世界模型 | 真实 LLM | 适用场景 |
|---|---:|---:|---:|---|
| `stable` | 是 | 否 | 否 | 默认产品、回归基线 |
| `research` | 是 | 已接受候选件 | 否 | 可审计模型研究 |
| `cognitive` | 是 | 已接受候选件 | 必须成功 | 冻结方案的真实供应商试验 |

认知严格模式禁止静默退回规则计划；研究代码仍可显式使用 fallback，但产物会标注来源，不能冒充真实 LLM 证据。

## 4. 模型生命周期

```text
dataset manifest
  -> TrainingJob(status.json + epochs.jsonl + latest.pt)
  -> candidate artifact
  -> dev calibration
  -> sealed validation
  -> decision artifact
  -> optional Studio adapter
```

世界模型与射门头已经解耦。射门头使用“冻结世界模型 latent + 去结果泄漏 action”，并绑定基础 checkpoint 的完整 SHA-256。样本数、进球数、Brier 相对物理先验、sealed-only 和 accepted 五个门槛缺一不可。

## 5. 运行可靠性

- 每个训练任务拥有隔离目录、原子状态、逐 epoch fsync 日志和原子续训点。
- Studio 只写 `STOP` 请求；训练器在 epoch 边界安全退出，不直接强杀进程。
- 控制面区分记录状态与观测状态；本机 PID 已消失的 `running` 会显示为 `stale`，并进入可恢复统计。
- LLM 配置在调用时从环境构造，并按配置隔离网关，避免测试或会话之间串用密钥与模型。
- Studio 对进程级环境变量适配加锁，并在比赛后完整恢复。

## 6. 当前证据结论

- M1 校准世界模型完成 18 场正式评测，结论为 `research_only_default_off`：可以研究使用，尚不能改变稳定默认行为。
- 独立射门头具备完整训练/验证/晋级架构，但尚未运行新训练，因此没有被接受的晋级产物。
- 真实 LLM pilot 已冻结流程和验收规则，但没有凭据时只做零成本 preflight，不产生真实调用证据。

这三项结论都意味着：产品结构已经统一，但研究能力仍受证据门控，不能用“代码存在”代替“效果成立”。

## 7. 用户操作面

```bash
python gfs.py studio init --name "My League" --mode stable --seed 42
python gfs.py studio status
python gfs.py studio match --home Brazil --away Argentina --fast
python gfs.py studio jobs
python gfs.py studio stop-job <run_id>
```

`studio init` 不会静默覆盖已有会话；只有显式传入 `--replace` 才会重置持久化历史。

## 8. V2.1 完成性修正

- 配置隔离从“修改进程环境并加锁”升级为 `ContextVar` 调用上下文；不同线程、异步任务和嵌入式调用互不污染。
- 训练目录使用操作系统文件租约，同一时刻只有一个写者；停止前强制保存当前 epoch，恢复时消费旧 `STOP`。
- 每次恢复拥有独立 `attempt_id`，数据内容和训练代码摘要都属于配置指纹，不能拿不同数据或代码继续旧优化器状态。
- Studio 在仿真前原子预约 run 与 seed；仿真完成后先写 `finalizing` provenance，再生成 JSON/HTML，任何失败都有持久记录。
- 稳定发布验证 `current.json -> release manifest -> 全部 artifacts`；研究 checkpoint 和独立射门头都由决策 SHA-256 绑定。
- `research`/`cognitive` 模式要求世界模型真正加载，加载失败不会静默退回稳定规则。
- wheel 明确保留 `src.*` 命名空间。代码包与数据工作区分离，通过当前目录、`GFS_PROJECT_ROOT` 或 `--base-dir` 连接。
- 所有训练入口登记在 `data/training/entrypoints.json`；Studio 只承认受管活跃入口，历史脚本仅用于冻结版本复现。

机器审计位于 `scripts/audit_architecture_v2.py`，验证依赖方向、任务状态机、发布/模型身份链、LLM provenance 和打包契约。当前代码工作树不是一次新的生产发布；发布晋级仍必须走独立 release manifest 流程。

`status` 是总览；`match` 生成组合 JSON 和 HTML；`jobs` 同时展示训练生命周期与世界模型、射门头、LLM 的决策状态。底层脚本保留给研发人员，但不再构成独立产品入口。
