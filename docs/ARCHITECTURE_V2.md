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
## 9. V2.2 战术实验与配对复盘闭环

Studio 的可玩入口现在分为两个有明确语义的体验，而不是暴露任意底层参数：

```text
原生观赛
  -> 球队原生战术画像
  -> macro_replay 稳定比分路径
  -> 单场组合报告

战术实验室（research / cognitive）
  -> 玩家选择主客队战术预设
  -> studio_user_intervention 完整战术向量
  -> physics_official 物理正式比分
  -> 单场组合报告
  -> 复用上一场同对阵、同快慢配置、同 seed
  -> 自动 paired comparison JSON + HTML
```

比赛方案由 `src/product/match_plan.py` 统一验证。未知预设失败关闭，不会静默变成 `balanced`；`stable` 禁止战术实验；原生观赛禁止复用种子。方案、比分路径、解析后的 seed 和推断边界属于任务幂等身份，改变任何一项都构成不同请求。

配对比较由 `src/product/comparison.py` 生成，并区分两层条件：

1. **结构资格**：不同比赛 ID、相同有序对阵、相同 seed、处理场明确引用基线、确实发生战术变化。结构不成立时不生成比较产物。
2. **归因资格**：相同快慢配置、相同 Studio 模式、两场均为物理比分、两场完整性通过、世界模型身份一致、无不受控认知供应商。检查失败时仍可生成描述性差值，但必须撤销战术归因资格。

单边改变只能解释为该侧在这一固定对阵和 seed 下的配对差值；双边改变只能解释为联合干预。任何单个配对都不是总体平均处理效应、显著性检验或普遍性能结论。`cognitive` 模式因真实供应商输出不受 seed 完全控制，自动降级为描述性比较。

每个合格处理场同时产生：单场 JSON、单场 HTML、比较 JSON、比较 HTML。Studio、处理场报告和比较报告互相提供受控 HTML 导航，用户可以完成“设置—运行—复用—比较—复盘”的单一产品旅程。

## 10. V2.3 固定预算战术研究闭环

单次同种子比较解决“能否形成受控差值”，但不能回答“这一差值在多个预先固定随机条件下是否稳定”。V2.3 在战术实验室之上增加 `TacticalStudyPlan`，把多次配对从零散比赛升级为一个预注册、可恢复、不可中途挑选结论的研究对象：

```text
研究模式 Studio
  -> 冻结对阵、单侧基线/处理战术、完整 seed 集合
  -> 固定预算（4..32 对），禁止运行后增删 seed
  -> 每个 seed 在同一会话租约中运行 baseline + treatment
  -> 运行期间仅公开 pairs_completed / fixed_pair_budget
  -> 全预算完成后一次性聚合
  -> primary: 干预侧 xG 差
  -> paired mean + sample SD + t interval + exact sign test
  -> 最终 JSON + HTML
```

该链路由 `src/product/tactical_study.py`、`src/product/tasks.py` 与 `src/product/web.py` 共同实现。协议要求恰好一侧战术变化、4–32 个唯一的 32 位非负种子、固定主要指标 `focus_xg_difference`、显著性水平 0.05、最小有意义效应 0.05；缺失配对使分析失败关闭，不允许以完整案例替代原定预算。进度文件不保存或公开中期效应，Web API 即使遇到篡改的进度或研究 ID 也只返回 `invalid_progress`，不会跨越工件目录或泄漏已完成子样本的结果。

最终结论只描述这一固定对阵和预注册种子集合中的平均配对差异。`beneficial`、`harmful` 或 `inconclusive` 都不等于跨球队总体效应，也不自动授予产品推广权。`stable` 与 `cognitive` 模式不能运行推断型战术研究；后者的外部认知供应商不受共享 seed 完全控制，只能留在描述性单场比较层。

固定预算完成后，研究 HTML 才开放逐 seed 证据下钻：每行同时链接基线比赛、处理比赛和共享 seed 配对页，并标记该 seed 是否支持总体方向。运行中进度不保存这些效果或下钻内容。恢复时，比较工件只能从当前 Studio 的 `matches/*.comparison.json` 读取；已完成结果也必须重新核对冻结计划、完整 seed 顺序、主要指标和 `promotion_authorized=false`，篡改结果不会被重新生成或发布。

Studio 首页提供统一证据资料库，把最近比赛、同 seed 配对和固定预算研究组织为同一浏览入口。比赛历史由会话清单提供，研究历史由后台任务清单提供，两者最多各读取最近 50 项；资料库只公开最小摘要与通过 HTML 白名单的工件 URL，不返回完整报告、研究 seed 列表或任何中期效应。持久化名称全部以 DOM `textContent` 呈现，不能作为 HTML 注入页面。

这里还必须区分两类证据：战术研究回答“改变战术后比赛输出是否系统变化”；世界模型动作采用协议回答“世界模型是否通过动作采用链路造成变化”。前者不能替代后者。世界模型机制协议仍保持独立固定预算和显式授权门，未正式执行前不得把运行观测写成机制成立。

## 11. V2.4 一键共享 seed 战术对决

手工执行“基线场—勾选复用—处理场”容易把错误的上一场当作基线。`PairedMatchPlan` 将单次配对冻结成一个显式请求：固定主客队、快速配置、共享 seed、基线战术和处理战术，并要求恰好一侧变化。后台 `paired_match` 任务调用 `run_paired_matches`，在同一会话租约内连续运行两场，最终只公开基线 HTML、处理 HTML 和配对比较 HTML，不把完整报告塞入任务记录。

```text
一键配对请求
  -> PairedMatchPlan（显式 seed + 单侧干预）
  -> 模式/就绪门禁
  -> 同一 session lease
     -> baseline physics match
     -> treatment physics match
  -> pairing eligibility checks
  -> baseline / treatment / comparison 三层复盘
  -> 统一证据资料库
```

`research` 模式可在全部资格检查通过后讨论这一固定对阵和 seed 的局部战术归因；`cognitive` 模式由于外部供应商输出不受 seed 完全控制，自动降级为描述性比较。单个配对永远不是总体效应或显著性检验。

配对任务以稳定 `task_id` 作为 `pair_transaction_id`，基线和处理比赛在会话记录与报告方案中分别写入 `pair_role=baseline|treatment`。中断后重排同一任务时，工作区会重新核对对阵、seed、快慢配置和两套战术：已提交基线通过 JSON/HTML 身份校验后直接复用，只补跑缺失处理场；若两场均已提交则零模拟返回原工件。处理场显式引用事务基线，不要求基线仍是最后一场，因此崩溃期间插入其他比赛也不会错配。相同事务 ID 搭配不同请求或重复角色会失败关闭。

证据资料库只为 `interrupted` 的配对任务显示“安全恢复”操作。`POST /api/v1/tasks/{task_id}/requeue` 要求 CSRF、原任务仍处于中断态和显式恢复原因；恢复继续使用原 task ID，遥测只记录低基数路由模板，不持久化 URL 中的任务身份。

## 12. V2.5 比赛动作轨迹复盘

单场报告现在把比赛引擎已有的传球与射门日志组织为产品内证据，而不是继续把底层 JSONL 当成孤立调试文件：

```text
micro match（唯一 match_id 作为 stage_name）
  -> outputs/ball_log/<fixture>_<match_id>.jsonl
  -> MatchReplay 防御性提取
     -> 工作区路径与对阵身份校验
     -> 文件/源事件/展示事件三层上限
     -> 非有限值拒绝、坐标裁剪、关键动作优先保留
  -> 组合比赛 JSON 的 replay 字段
  -> 严格 CSP 下的纯 SVG + CSS 轨迹浏览器
     -> 全部 / 传球 / 射门 / 主队 / 客队
     -> 有界关键事件表
```

Studio 的三个运行模式都显式启用 JSONL 球日志并关闭文本主格式，比赛执行器同时接收真实工作区根目录和唯一比赛 ID，因此不同比赛不再覆盖同名日志，安装后的代码目录也不会误收产品工件。回放提取器只接受 `<workspace>/outputs/ball_log/*.jsonl`，上限为 4 MiB、2,000 个源事件和 240 个展示事件；超界、对阵不符、路径逃逸、编码错误或无有效动作都会失败关闭为明确的 `reason`，但不伪装成比分完整性失败。

该能力的严格产品名称是“动作轨迹复盘”：目前只有球的传球落点和射门到球门方向，没有 22 名球员连续位置、摄像机画面、时间连续插值或视频语义。报告必须持续展示这一边界。它提高的是单场可检查性、可玩性和世界模型/战术结果的情境解释能力，不会把描述性轨迹自动升级为动作采用机制证据或战术因果证据；后两者仍分别受独立固定预算协议约束。

## 13. V2.6 世界模型决策—球场动作身份贯通

仅凭球队、动作类型和相近时间把世界模型决策与球场日志拼接，会在同一 tick 多动作、日志截断或历史数据中产生伪对应。V2.6 因此不使用时间近邻推断：动作采用层生成的稳定 `opportunity_id` 在传球或射门真正执行时直接进入球日志，产品关联器再执行一对一四重核验。

```text
action opportunity
  -> opportunity_id
  -> shared-draw policy sample（baseline / adjusted / actual）
  -> pass or shot execution
  -> ball-log event.wm_action_opportunity_id
  -> bounded replay extraction（优先保留带身份事件）
  -> exact identity + action + team + timestamp
     -> direct_runtime_identity_match
     -> 或明确的 collision / mismatch / missing / truncated 状态
```

身份键必须在动作记录侧和球日志侧同时唯一；动作、球队或时间任一不一致都失败关闭。`hold` 没有球轨迹，保留为 `decision_record_only`；`cross` 从 V3.64 起记录独立球轨迹、落点、争顶结果和状态快照。旧日志没有身份键时标记为 legacy，不回退到时间猜测。报告中的世界模型决策行可以跳到通过四重核验的 SVG 轨迹，球场则提供 `WM-linked`、`WM changed + observed` 和传中专项筛选。稳定模式或无动作采用记录的比赛不显示无意义的世界模型筛选。

“直接身份匹配”只证明两份运行记录描述的是同一次动作，不等于世界模型造成了比赛结果变化。即使某条记录同时满足 `policy_changed_action=true` 与直接轨迹匹配，也只是局部共享随机数反事实与实际执行的可审计贯通；总体机制结论仍必须等待独立固定预算动作采用研究获得显式授权并完整执行。

## 14. V2.7 严格 CSP 动作序列播放器

动作复盘从“全部轨迹静态叠加”升级为有顺序的手动播放器，但不放宽比赛工件的安全策略。工件响应仍是 `default-src 'none'` 且没有 `script-src`；播放器完全由原生 radio 状态、生成自整数帧号的 CSS 选择器和 SVG 组成，不执行持久化文本，也不加载外部资源。

每份报告最多暴露 240 个已防御性归一化的动作帧，并提供：完整轨迹概览、横向时间轴任意跳转、上一动作、下一动作、末帧返回概览、当前动作文字详情、历史累计轨迹、未来轨迹隐藏和当前动作高亮。动作类型、球队与世界模型身份筛选仍可组合使用。键盘用户在同一个 radio group 中使用方向键切换帧，当前时间标记具有显式焦点样式；窄屏下动作详情与前后按钮自动重排。

```text
bounded replay events
  -> overview frame
  -> step 1 ... step N（N <= 240）
     -> previous / next / direct timestamp jump
     -> prior trajectories as context
     -> current trajectory emphasized
     -> future trajectories hidden
     -> optional WM-linked / WM-changed badge
```

这仍是离散传球和射门序列，不是按真实时间自动播放的视频，也不是连续的 22 人位置动画。无脚本设计优先保证可移植、可离线审计和严格 CSP；若未来加入连续球员追踪，必须建立独立的数据规模、隐私、插值真实性和渲染性能契约，不能用当前动作点列冒充完整比赛动画。

## 15. V2.8 同 seed 双球场同步复盘

配对比较页现在不仅展示汇总差值，还把基线场和处理场的动作回放放在同一共享比赛时钟上。同步只表示“截至相同比赛时间，两场各自发生了哪些已保留动作”；它不寻找跨场最近动作、不声明两个动作互为对应，也不改变原有配对资格检查。

```text
validated same-seed comparison
  -> baseline bounded replay（<= 240 actions）
  -> treatment bounded replay（<= 240 actions）
  -> union of retained match-clock timestamps
     -> <= 240 shared frames
     -> shots / goals / saves / interceptions / WM-changed timestamps first
  -> one CSS state controls two SVG pitches
     -> cumulative baseline trail
     -> cumulative treatment trail
     -> current actions highlighted independently
     -> “no new retained action” when only the other side acts
```

每侧事件集合必须是数组，事件必须具有有限时间、合法主客队身份、`pass|shot` 类型和可裁剪二维坐标；非数组、损坏行和未知球队失败关闭或被计数忽略。每侧最多消费 240 个事件，共享时点最多 240 个；超过时点预算时优先保留射门、关键结果和 `WM changed`，其余时点均匀抽样。渲染器不信任持久化 `frames`，而是从已归一化的两侧事件重新构造时间轴，防止篡改可见计数、CSS 帧号或注入文本。

双场页面提供全部、仅射门、`WM changed` 筛选，横向共享时间轴、上一/下一时点、末帧返回概览、两场当前动作详情和描述性动作计数差值。页面继续遵守无脚本工件 CSP，并在窄屏上把双球场和两侧当前动作改为纵向布局。任一侧回放不可用时，只有播放器失败关闭；指标比较、结构配对和归因资格保持各自原有结论。

同步播放器中的 `event_correspondence_authorized=false` 与 `causal_claim_authorized=false` 是固定字段。即使配对满足局部战术归因资格，轨迹层也只帮助解释这一固定对阵和 seed 下的行为差异；它不把视觉相似、时间接近或某次射门差异升级为事件级因果链，更不替代固定预算多 seed 研究。

## 16. V2.9 可恢复赛季与长期球队状态

单场、同 seed 配对和固定预算研究分别解决观赛、局部战术比较和受控推断，但不会自然形成长期可玩的产品。V2.9 增加 `SeasonPlan` 与 Studio 赛季状态机，把比赛、球队连续状态、积分榜和复盘工件组织为一个持久竞争过程；研究比赛继续保持无副作用，只有显式赛季链路允许写入长期状态。

```text
SeasonPlan（4..16 unique teams, 1|2 legs, seed, focus team）
  -> deterministic circle-method schedule
  -> stable season_id / matchday / fixture_id
  -> persistent season_matchday background task
     -> fixture marked running
     -> Studio match + continuity_id=season_id:fixture_id
     -> report committed
     -> fixture result committed
     -> standings recomputed from completed fixtures
  -> idempotent matchday recovery
  -> next matchday
```

赛程采用确定性的 circle method。奇数球队自动轮空；同一比赛日任何球队最多出场一次；单循环每个无序对阵恰好一次，双循环反转主客场恰好两次。赛季最多 16 队，因而最多 240 场，避免无界会话和页面负载。积分榜不维护可重复累加的独立计数，而是每次从已完成且通过契约校验的 fixture 重算，排序固定为积分、净胜球、进球数、球队名，防止重试造成重复计分。

跨比赛状态由 `TeamSquadCarryover` 保存球队疲劳 EMA、士气、媒体压力、球员出场、状态、伤病与停赛。持久化现在使用文件租约、保留既有球队的合并写入和同目录 `fsync + os.replace`，不能再由一场两队比赛抹掉其他球队。只有实际获得正分钟的球员累计出场；旧伤停先消费当前 fixture，新伤停和新红牌在赛后生效。比赛日结束后执行一次确定性疲劳/压力恢复。

比赛结算和比赛日恢复分别具有稳定事务键。若进程在球队状态已落盘、赛季 fixture 尚未提交时中断，同一 `season_id:fixture_id` 重试会跳过已应用结算；若比赛报告已经进入 Studio 会话但赛季记录尚未认领，恢复流程先执行报告身份与 JSON/HTML 可用性校验，再直接绑定已有 match，不重复模拟。比赛日恢复同样记录 `season_id:recovery:matchday`，崩溃重试不会提供双倍休息。重复完成报告、赛程身份变化、非法比分或赛季/比赛日任务身份冲突均失败关闭。

Web 端通过 `POST /api/v1/seasons` 创建赛季，通过持久化 `season_matchday` 后台任务推进比赛日。后者使用自动稳定幂等键，避免双击、刷新或相同请求重放生成重复任务；后台 worker 将冻结的 season 和 matchday 身份传回工作区核验。Studio 赛季中心只用 DOM `textContent` 构造积分、赛程与状态，比赛复盘链接继续通过 `outputs/studio/*.html` 白名单映射，不信任持久化 HTML。

赛季积分与状态只描述这一固定模拟赛季，不估计现实球队实力，也不构成世界模型或战术机制证据。研究/认知模式中的世界模型动作采用仍可在赛季单场内被观察和复盘，但任何总体机制结论仍受独立固定预算协议与显式授权门约束；本阶段没有启动训练，也没有执行正式动作采用机制研究。

## 17. V3.0 经理决策与跨比赛取舍闭环

V3.0 在可恢复赛季上增加 `ManagerDecision`，使玩家在焦点球队每场比赛前做出一项冻结决策，而不是只观看自动赛程。决策由战术与轮换两部分组成；它们进入同一场比赛的实际计算链路，并在赛后通过疲劳持久化影响后续比赛。

```text
next focus-team fixture
  -> ManagerDecision(team, tactic, rotation)
  -> validate focus team + next fixture + unstarted state
  -> increment season revision
  -> season_matchday task freezes season_id + matchday + revision
  -> MatchPlan(experience=season_manager, context=season)
     -> tactic preset changes the real tactical vector
     -> rotation changes effective status and fatigue load
     -> continuity state changes the base effective status
  -> macro score prior + micro physics receive the same effective status
  -> report records decision and realized gameplay effects
  -> fatigue/morale/injury settlement persists into the next fixture
```

`ManagerDecision` 只允许作用于焦点球队的下一场未开始比赛。缺少决策时，整个比赛日会在任何 fixture 增加 attempts 或启动模拟前失败关闭；不能越过下一场提前写入后续决策。比赛开始后决策冻结。每次设置或替换决策都会递增赛季 `revision`，后台任务把该版本写入身份契约；排队后若玩家修改决策，旧任务会因 revision 冲突而拒绝执行，不能用旧选择推进新状态。

轮换提供三档明确取舍：`strongest` 不施加即时阵容代价，`balanced` 施加 1.25 状态点代价，`rotate` 施加 3.0 状态点代价。与此同时，轮换强度写入真实 `rotation_aggressiveness`，由战术效果公式降低赛后疲劳负荷。比赛双方的统一有效状态为：

```text
base_status = continuity ? agent.get_effective_status() : agent.status_score
effective_status = max(12, base_status + tactical_status_delta - rotation_penalty)
```

该 `effective_status` 同时传给宏观 Poisson 比分先验和微观物理引擎，避免“UI 有选择、报告有字段、比赛行为却不变”的伪闭环。赛后结算接收同一战术效果产生的 `fatigue_load_multiplier`；因此轮换既影响本场实力，也影响下一场的疲劳基线。焦点球队之外仍保持原生战术和无经理轮换覆盖。

Web 端通过 `POST /api/v1/seasons/decision` 提交下一场决策。赛季中心显示下一对阵、战术、轮换和已冻结选择；存在焦点球队但尚未提交决策时，“推进比赛日”不可用。决策 API、赛程状态和任务版本号均为结构化 JSON；页面继续用 DOM `textContent` 呈现持久化值。

单场报告新增 `management` 层，分别保存请求决策和实际生效结果，包括基础状态、最终有效状态、轮换代价及疲劳倍率。该层的边界固定为游戏内干预说明，不构成现实球队表现估计、战术因果结论或世界模型机制证据。`season_manager` 方案只能在工作区的 season context 中运行，独立比赛 API 不能伪造该体验。

本阶段没有启动训练，也没有执行正式世界模型动作采用机制研究。V3.0 的验证范围是产品行为闭环、任务身份、崩溃恢复和游戏状态连续性；世界模型总体机制结论仍受独立固定预算协议及显式授权门约束。

## 18. V3.1 冻结阵容与球员级赛季管理

V3.0 的 `rotation` 已经改变球队级有效实力和赛后疲劳，但没有决定哪 11 名球员实际出场。V3.1 将经理决策继续向下贯通到现有 roster、微观球员状态、换人引擎和赛后分钟结算，使“轮换”从抽象倍率升级为可见、可选、可恢复的球员名单。

```text
raw team roster + TeamSquadCarryover
  -> apply injuries / suspensions / form / condition
  -> bounded squad catalog + roster_fingerprint
  -> manual lineup OR deterministic automatic lineup
     -> exactly 11 starters
     -> exactly one trusted goalkeeper
     -> at most 12 bench players
     -> no duplicate, unknown or unavailable player
  -> freeze inside ManagerDecision + increment season revision
  -> pre-match fingerprint revalidation
  -> roster roles: starter / bench / reserve / suspended
  -> existing micro squad builder + substitution engine
  -> actual player minutes and match feedback
  -> next-fixture carryover
```

阵容领域契约位于 `src/simulation/lineup.py`。它属于比赛连续性领域层，产品层只负责冻结和编排，避免 simulation 反向依赖 product。`LineupSelection` 保存首发、替补、来源和 64 位 roster 状态指纹。手动名单必须恰好 11 名首发、最多 12 名替补、所有球员唯一且属于当前球队；首发必须恰好包含一名可信 `role=GK` 的门将。伤病未恢复、停赛或 availability 不合格的球员不可选择。名单一旦写入比赛决策，就与战术、轮换和赛季 revision 共同构成任务身份。

未手动选择时，系统按当前阵型逐位置确定性生成名单。`strongest` 主要使用能力、状态和可用性；`balanced` 同时惩罚历史分钟负荷；`rotate` 进一步提高分钟负荷惩罚并为原替补提供有限轮换优先级。位置选择先匹配原生角色，再使用有界的足球位置兼容表，且不会用普通场上球员填补门将。相同 roster 状态与轮换策略必然生成相同名单。

状态指纹覆盖球队、阵型、球员 ID、位置、availability、condition 和 form。工作区提交决策时冻结指纹；比赛准备层应用同一长期状态后重新计算。任何外部伤停改写、球员缺失或状态漂移都会在微观比赛开始前失败关闭。球员名单只允许在 `continuity=True` 的 roster 链路中使用，独立单场入口不能悄悄忽略名单参数。

阵容应用后，只有冻结的 11 人获得 `starter`，只有冻结替补获得 `bench`，其他可用球员统一成为 `reserve`；roster loader 不再把所有非首发自动加入替补池。因此换人引擎只能使用经理实际选择的替补。微观球员追踪器记录真实在场分钟，赛后 carryover 只为实际出场球员累计出场与分钟负荷。

Studio 赛季中心提供当前球员目录、伤停状态、三套自动建议和手动名单选择。所有球员名称通过 DOM `textContent` 呈现，动态 select 具有可访问名称；手动提交前检查 11 人首发与替补上限。比赛 HTML 新增经理面板，展示战术、轮换、有效实力变化、疲劳倍率、首发和替补，并继续声明单场游戏效果不构成现实表现或战术因果证据。

`scripts/verify_lineup_catalog.py` 对冻结的 48 份 roster 执行零比赛、零训练、零外部调用审计。当前 46 队支持完整球员级管理；Algeria 与 Iran 的源 roster 没有可信门将记录，且门将能力只是统一默认值，因此被明确标记为 `roster_missing_goalkeeper`。系统不会伪造角色：这两队仍可使用球队级战术和轮换继续赛季，但不能声称拥有球员级冻结名单。真实 roster 存在门将但因动态伤停无法组成合法 11 人时不会降级，而是失败关闭。

本阶段没有修改冻结 roster 数据、没有启动训练、没有执行正式世界模型机制研究。阵容审计证明的是名单构建完整性和产品链路覆盖，不证明现实阵容准确性、比赛预测有效性或球员选择因果收益。

## 19. 世界模型动作采用链路 V2

V2 将世界模型从“计算了动作价值但几乎不改变执行”修正为两级、单点、有界的真实动作控制链。验证质量只使用一次，作为动作头的硬授权门；通过门后，当前观测覆盖率、在线校准信任和剔除验证质量重复项后的模型分歧共同决定本次干预预算。未经验证的射门、横传和持球动作头继续失败关闭，不能借用传球头的质量。

```text
checkpoint action-head validation
  -> hard authorization gate
  -> current-state confidence + online trust + model disagreement
  -> bounded high-level probability controller (pass vs continuation)
  -> shared-uniform baseline and world-model samples
  -> if pass: score the exact executable receiver/target candidates
  -> bounded target-policy blend
  -> execute the selected encoded pass candidate
  -> record action and target counterfactual identities
```

高层控制器是唯一的动作类别采用点。模型效用差只作为可审计证据，不再先改一次效用、再改一次概率。所有不可执行动作在采样前被置零并重新归一化，不能先抽到非法动作再回退为传球。`recommended_action` 表示模型实际推高的动作，而不是混合后绝对效用最大的原策略动作；只有存在非零策略信号或效用变化时才计为 influenced opportunity。

传球目标层直接评估 PassingEngine 已生成的真实候选，包括接球队员、传球类型、落点和物理成功先验。基础策略与世界模型策略使用同一个均匀随机数，记录实际候选、无世界模型反事实候选、总变差距离和是否改变目标。正式机制数据同时导出高层动作变化和传球目标变化；`passing_engine.py` 已加入协议代码身份，避免执行实现发生变化而协议仍错误显示一致。

当前封存 checkpoint 只授权传球头（质量 0.176654，高于 0.15 门槛）；射门头质量 0.005400，保持关闭。一次无比赛诊断中，高层传球概率改变约 0.95 个百分点；实际四候选目标诊断产生不同评分和非零概率变化，但模型价值跨度仍小。该结果证明代码链路不再静默失效，不证明机制总体已通过，更不证明比赛结果改善。没有启动训练，也没有执行 24 场预注册机制研究。

## 20. V3.2 实时比分驱动的临场管理

V3.2 把赛前经理决策延伸到比赛中的真实状态机。普通射门原本已经通过 `GOAL_SCORED` 事件即时更新比分；本阶段补齐空中球/头球进球事件，使它们同时进入比分、球员统计、情绪、战术压力、认知显著性和世界模型观测，避免最终统计有进球而比赛中状态仍未变化。

`InMatchPlan` 最多包含 5 条分钟递增且分钟唯一的指令。每条指令在指定分钟首次到达时只评估一次，条件为 `always|trailing|drawing|leading`，动作可以是战术切换、冻结名单中的指定换人，或二者同时。条件不满足时记录 `skipped`，不会在稍后比分变化时偷偷触发；红牌、球员状态冲突或换人名额耗尽时记录 `failed` 和稳定原因，不自动替换成其他球员。

```text
ManagerDecision + frozen lineup + InMatchPlan
  -> season revision / task identity
  -> live goal events update score
  -> exact-minute condition evaluation
  -> validated tactical preset and/or exact substitution
  -> subsequent physics, emotion, cognition and world-model ticks
  -> applied / skipped / failed audit
  -> report panel + replay timeline annotation
```

包含指定换人的经理球队关闭随机自动换人，保证系统不会提前消耗计划球员或名额；对手仍保留原自动换人逻辑。指定下场球员必须属于冻结首发，上场球员必须属于冻结替补，同一球员不能出现在两次计划换人中，门将与非门将不能错误互换。战术变化直接替换实时教练向量，影响后续动作选择，而不是只写入报告字段。

Studio 提供 3 条可编辑规则，支持分钟、比分条件、战术以及来自当前冻结首发/替补的球员选择。所有动态球员名称继续通过 `textContent` 构造。赛后经理面板和轨迹回放均显示临场事件的分钟、当时比分、请求、状态和原因。该功能是模拟产品玩法，不构成现实战术因果证据；本阶段没有训练，也没有执行正式世界模型机制研究。

## 21. V3.3 统一比赛日指挥台

V3.3 不再让赛前决策、赛季推进、积分榜和赛后报告作为四块互不解释的功能并列出现。`matchday_command_center` 是从冻结赛程、已持久化赛果和经理决策即时重建的只读模型，成为 Web 产品当前比赛日状态的唯一权威来源；它不持久化第二套进度，也不修改比赛结果。

```text
persisted season + frozen fixtures + completed scores
  -> validate_season_state
  -> matchday_command_center
     -> current phase and exactly one primary action
     -> current/next managed fixture and table context
     -> frozen-decision readiness
     -> five-match form and last managed result
     -> safe report link resolved by the Web artifact boundary
  -> accessible matchday journey
```

阶段明确区分 `decision_required`、`ready_to_advance`、`ready_to_resume` 和 `season_complete`。指挥台显示关注球队与对手的当前排名和积分、主客场、近期状态、决策冻结情况、上一场比分与积分收益，并将完整审计报告作为赛后复盘入口。展示使用 DOM `textContent`，复盘链接只能来自 Web 层已经校验过的工作区 HTML 工件，不能直接使用状态中的任意路径。

奇数球队赛季的轮空被建模为本轮无需经理决策，因此未来场次尚未提交决策不会再错误阻塞当前比赛日。若关注球队已经完成本轮、但同轮其他比赛失败或中断，则阶段为 `ready_to_resume`，不会误称为轮空，也不会重复执行已完成比赛。这个区分与后端原有幂等恢复身份保持一致。

该层只陈述当前模拟赛季内部的排名、状态与赛果，不把积分形势包装为现实球队实力预测。自动可访问性验证覆盖原子实时播报、键盘焦点、移动端单列、强制颜色、对比度和真实 HTTP 页面；它仍不替代浏览器可访问性树、屏幕阅读器或目标用户测试。

## 22. V3.4 证据约束的经理情报与归因

V3.4 让比赛日指挥台从状态导航升级为决策工具，但拒绝用模板或语言模型猜测“最佳战术”。赛前情报只读取已经进入比赛机制的事实：持久化球队疲劳、士气、媒体压力、伤停与停赛，当前 roster 的可选人数、阵型、球员出场负荷和状态，以及积分榜位置。每条风险提示同时保存阈值命中的直接证据和可操作问题，不输出胜率或战术排名。

球队状态来源分为 `persisted_squad_carryover` 与 `default_simulation_baseline`。比赛日情报进一步区分双方直接持久化状态、单边持久化与单边基线、双方 roster 默认基线，以及部分球队状态，避免把默认值描述成历史观测。轮换区域展示的是实际引擎合同：`strongest|balanced|rotate` 的 rotation level 为 `0|0.5|1.0`，显式状态代价为 `0|1.25|3.0`；它是玩法权衡，不是比赛结果预测。

```text
validated season + manager/opponent roster catalogs
  -> bounded team-condition facts
  -> evidence coverage grade
  -> deterministic risk thresholds and decision prompts
  -> no suggested tactic / no win probability

last managed fixture + workspace-contained JSON report
  -> path, suffix, size and match identity checks
  -> frozen decision + direct engine effects
  -> applied / skipped / failed in-match instructions
  -> observed score marked descriptive-only
  -> causal_outcome_attribution = false
```

赛后归因只把两类内容称为直接证据：引擎实际应用的状态变化、轮换代价和疲劳负荷倍率；临场指令的实际执行、跳过或失败。比分与胜负可以复盘，但一次运行不能证明经理决策导致赛果。若报告越出工作区、不是 JSON、超过 32MB、比赛身份不匹配或嵌套结构损坏，整个赛季页面继续可用，归因区域明确降级为证据不可用。

Web 端所有球队名、风险证据、决策提示和归因字段继续通过 DOM `textContent` 写入。风险颜色同时保留文字严重度和列表语义；自动可访问性验证覆盖真实 HTTP 表面。本阶段没有训练、没有外部提供方调用，也没有执行正式世界模型机制研究。

## 23. V3.5 有界的多赛季经理生涯

V3.5 把单个可恢复赛季扩展为连续的经理生涯，但不引入另一套平行进度。开季时必须冻结一项经理目标：夺冠、上半区或指定积分；积分目标必须处于该赛制理论可达范围内。目标与关注球队、参赛队和赛程共同进入持久化赛季计划，赛季进行中不能被改写。当前目标面板只依据已完成比赛和剩余理论积分展示 `currently_meeting`、`in_progress` 或 `unreachable`，不把中途排名包装成最终预测。

```text
frozen SeasonPlan + persisted fixtures + ManagerDecision records
  -> derived manager season profile
     -> objective progress and mathematical reachability
     -> W/D/L, points and current/final position
     -> tactic, rotation and lineup-source usage frequencies
     -> one journal entry per completed managed fixture
  -> all fixtures completed
  -> immutable season archive
  -> stable next season identity
  -> existing cross-match squad continuity carries forward
```

赛季日志只从已经持久化的比赛、冻结决策和安全报告引用重建，不维护第二份可漂移的统计状态。每条记录包含对手、主客场、比分、积分收益、战术、轮换、阵容来源、计划临场指令数以及报告入口。战术或轮换的“常用”仅表示模拟生涯中的观测频率；它不是有效性排名，也不证明某项决策导致了胜负。赛季完成后，目标状态才收敛为 `achieved` 或 `missed`。

开启下一赛季是一项显式操作。存在活动赛季时，普通创建继续失败；只有全部赛程完成且请求明确携带 `start_next`，系统才会先校验并归档旧赛季，再创建新赛季。赛季 ID 单调稳定并检查历史冲突。球队的长期疲劳、士气、伤停、停赛和球员负荷继续使用既有跨比赛状态，因此新赛季不是暗中清空的独立沙盒；赛程、积分、冻结决策和赛季目标则从新计划重新开始。

工作区最多保留 12 份完整赛季档案，并单独记录历史总数、当前保留数、上限和是否发生截断。读取时会验证档案 ID 唯一性、计划、赛程规模、最终积分榜、经理身份、目标终态以及日志比赛身份；损坏档案使状态读取失败，而不是静默展示伪造履历。Web 层再次把日志中的报告路径限制为工作区内受支持的 HTML 工件，历史数据不能借机生成任意文件链接。

Studio 在赛季结束后把创建表单切换为“归档本季并开始下一赛季”，同时展示本季目标结果、完整记录和有界历史摘要。所有动态名称和日志字段继续经 `textContent` 渲染。本阶段只完成产品代码、持久化契约和自动验证，没有启动训练、外部服务调用或正式世界模型机制研究。

## 24. V3.6 有后果的经理合同与董事会评价

V3.6 让赛季履历产生下一赛季的游戏后果，但不新增一份可与赛季档案漂移的经理数据库。`manager_career_profile` 按顺序折算最近 12 份已通过完整性校验的赛季档案；当前信任、声望、任职状态和合同要求全部是可重建读模型。详细历史发生截断时，界面明确称其为滚动窗口，不声称这是未截断的终身评价。

单季董事会评价是冻结的游戏规则。目标达成计 `+18` 信任，未达成计 `-20`；最终排名按联赛首尾之间的线性位置贡献 `+8` 到 `-8`；每场经理决策完整记录贡献 `+4`，覆盖至少 80% 贡献 `+2`，否则不加分。单季信任变化限制在 `[-28,+30]`。声望使用目标 `+6/-5` 与排名贡献的一半，并限制在 `[-9,+10]`。董事会信任初始为 60，职业声望初始为 50，二者始终限制在 `[0,100]`。

```text
validated final table + frozen objective + complete decision journal
  -> reproducible single-season board review
  -> confidence delta + reputation delta + component evidence
  -> fold retained archive window in season order
  -> secure >= 75
     stable >= 45
     under_review >= 20
     dismissed < 20
  -> next-season contract validation before any session write
```

换队时，新董事会信任重置为 60，滚动职业声望继续保留；当前任命计入任职次数。处于观察期的经理如果选择同队积分目标，目标不得低于上一季积分加 3 分，并受新赛制理论最高分约束；夺冠和上半区目标仍可选择。信任低于 20 时，经理失去同一俱乐部的续任资格，但可以在下一赛季选择另一支参赛球队继续职业生涯。该规则创造失败成本和换队路径，不宣称模拟结果代表现实就业判断。

赛季全部完成后，Studio 立即从尚未归档的当前赛季计算待确认评价，展示归档后的预计信任、声望、任职状态和合同条件；此时持久化职业历史仍未改变。用户显式开启下一赛季时，工作区在同一个锁内重建旧赛季档案、评价和职业合同，先验证新计划，再一次性写盘。不合格目标、被解雇后试图留队或被篡改的档案都会失败关闭，且不会留下部分归档或错误历史计数。

档案校验不只检查字段类型，还要求最终积分榜、经理排名与积分、目标终态、完整比赛数、逐场结果、胜平负、积分恒等式和董事会评价互相一致。旧 V3.5 档案可以在读取时从原始证据补算评价；若已存评价与补算结果不同则拒绝使用。Web 端只通过 `textContent` 呈现动态合同信息，并在提交前提示明显冲突；后端仍是最终权威。本阶段没有训练、比赛执行、外部服务调用或正式世界模型机制研究。

## 25. V3.7 有界的联赛对手记忆与真实战术响应

V3.7 解决长期赛季中对手始终使用静态原生体系、经理可以无成本重复同一策略的问题。对手准备不是语言模型提示或报告标签，而是在经理提交下一场决策的同一事务中冻结，并直接成为对手侧 `MatchPlan` 的战术预设。比赛引擎继续通过既有 `apply_locked_tactical_preset` 应用完整战术向量，因此响应会影响后续实际动作选择、物理比赛和比分路径。

对手只能读取同一持久化赛季中、当前受管比赛之前已经完成且保存了经理决策的比赛。当前场经理刚提交的战术不进入观察集合，未完成、失败、未来比赛、单场实验和其他工作区记录也不进入。至少需要 2 场历史观察，且主导战术占比达到 60%，才允许覆盖对手原生体系；样本不足、模式分散、经理主要使用原生体系或规则不支持时一律回退 `team_identity`。

```text
earlier completed managed fixtures only
  -> frozen tactic counts and dominant share
  -> minimum observations = 2
  -> minimum dominant share = 0.60
  -> bounded_opponent_response_v1
  -> freeze with ManagerDecision + increment season revision
  -> opponent-side season_manager MatchPlan tactic
  -> real tactical preset application
  -> match report + season journal + archive audit
```

响应映射是显式版本化的游戏设计：`balanced -> gegenpress`、`tiki_taka -> low_block_counter`、`gegenpress -> direct_vertical`、`counter_attack -> balanced`、`low_block_counter -> tiki_taka`、`direct_vertical -> low_block_counter`。这些映射用于制造可理解的赛季博弈和鼓励风格变化，不声称对应战术在现实中具有已证明的克制效果，也没有通过当前项目数据学习得到。

比赛日指挥台在提交决策前展示根据当前历史可复算的准备状态、观察场数和将使用的响应；提交后同一内容标记为已冻结。赛后复盘显示实际响应战术与证据样本数，逐场经理日志继续保存该审计记录。经理仍可在当前比赛选择与过去主导模式不同的战术，因为对手不会偷看当前选择；这形成可玩的风格伪装与变化空间，而不是让系统自动读取用户答案。

恢复路径要求已完成比赛记录中的 `opponent_preparation` 与冻结赛季状态完全一致，否则拒绝把旧报告接回赛程。活动赛季通过原始赛程和更早决策重新计算每一场准备；归档后则从有序经理日志使用同一个构造函数重建并逐项比较，避免另写一套漂移规则。任何响应战术、样本、主导占比、对手身份或策略版本篡改都会失败关闭。本阶段没有训练、正式机制研究、外部服务调用或为了验证而执行比赛。

## 26. V3.8 Club resources and long-horizon strategy

V3.8 extends the manager career from isolated match choices to a frozen season-level resource trade-off. Every managed season allocates exactly 6 points across recovery, medical, and sports science, with each area bounded to 0..4. The allocation is part of `SeasonPlan.manager_resources`; it cannot change during the season and travels with the season, match plan, runtime report, and recovery transaction identity.

```text
frozen ClubResourcePlan (budget = 6, each area = 0..4)
  -> recovery: next-matchday rest units = 1.0 + 0.12 * points
  -> medical: injured-player recovery credit = 0.15 * points / matchday
  -> sports science: future fatigue load factor = 1.0 - 0.04 * points
  -> current-match status bonus = 0
  -> persisted carryover + audited report + next fixture briefing
```

Recovery applies only after the whole matchday completes and only to the managed club; every other team retains the 1.0 baseline. Stable per-team matchday transaction IDs make retries idempotent. Medical credit accrues only for players who are currently injured: each full credit removes one extra injury match, fractions carry forward, healthy players cannot pre-bank credit, and suspensions never change.

Sports science scales only the tactical fatigue load written to future continuity state. It does not alter the effective status supplied to the macro score prior or micro physics engine, and it does not alter current-match xG. A non-1.0 club factor is rejected outside continuity play. In a managed season, the runtime validates the frozen support identity, recomputes the allocation effects, requires the exact manager-side factor, and requires the opponent factor to remain 1.0.

The match plan records the supported team, complete plan, fixed budget, pure-function effects, and the explicit zero current-match status bonus. The report repeats the same evidence under `management.club_support` and records the actually used `club_fatigue_load_factor`. The post-match read model reparses the plan, recomputes every effect, and compares the planned factor with the runtime factor. Tampering makes direct evidence unavailable; a legacy report without resource evidence is shown as a compatibility downgrade instead of inventing an allocation.

Studio validates the 6/6 budget live, shows the frozen allocation and exact long-term effects in the matchday command center, and shows the audited resource execution in the debrief. This closes one product loop: season-level resources, pre-match lineup and tactics, in-match instructions, and post-match carryover all influence the state visible before the next fixture. These are bounded simulation mechanics, not claims about real club investment or tactical causality. This stage starts no training, formal world-model study, external service, or validation match run.

## 27. V3.9 Replayable squad building and recruitment

V3.9 makes squad construction a real cross-season mechanic instead of a detached transfer screen. Recruitment opens only after a completed season and is submitted atomically with creation of the next season. A managed club receives a deterministic fictional market with 5 candidates, a fixed budget of 8 game credits, and at most 2 moves. Every move is an inseparable incoming-candidate and outgoing-player pair, so squad size cannot inflate through repeated windows.

```text
completed season + next season index + current effective roster
  -> market_id bound to season index and club identity
  -> roster identity hash
  -> weakest formation-role ordering
  -> 5 deterministic fictional candidates (costs 3, 3, 4, 5, 6)
  -> choose 0..2 incoming/outgoing pairs within 8 credits
  -> replayable squad transaction
  -> atomic product session + next SeasonPlan
  -> effective roster used by lineup, substitutions, micro simulation and settlement
```

Candidate IDs, names, roles, ages, abilities, conditions, costs and quality summaries are pure functions of the market ID, club identity and prior effective-roster hash. They do not use a provider, hidden random state, real scouting data or real transfer prices. The market therefore remains stable across page refreshes, but changes after an earlier valid transaction changes the effective roster. A client cannot invent a candidate, reuse a candidate, release the same player twice, exceed the budget, release a player outside the current squad, grow the squad, or leave the club without a goalkeeper.

Base files under `data/rosters` remain immutable. The authoritative `squad_registry` lives in `data/persistence/product_session.json`, in the same atomic write as season archive and next-season creation. Each transaction stores its season, plan, budget, prior/result roster identities and full incoming/outgoing evidence. On every workspace read, all registered clubs are replayed from their base rosters; unknown fields, duplicate markets, missing base rosters, hash drift, changed costs, changed player evidence or disagreement between the active/archive season and registry fail closed.

The effective-roster loader is shared by lineup catalogs, match preparation, post-match settlement, world/tournament roster dynamics and the fallback squad factory. Recruitment therefore changes the players that can be selected, the automatic lineup quality ordering, the bench and substitution pool, and the player abilities entering micro simulation. It is not a product-only overlay. Before a match, carryover state is reconciled to the effective roster: released players and their injuries or suspensions stop affecting team availability, while a signed player receives a new bounded carryover record.

Studio exposes a read-only, privacy-normalized market route. The next-season form contains two accessible incoming/outgoing rows, live budget and completeness checks, stable candidate quality/cost labels, and optional zero-move continuation. The current manager profile and retained season history show the exact audited moves and budget spend. Dynamic text continues to use DOM `textContent`; team path components are collapsed to a telemetry route template.

This mechanic links the career loop across time: board/contract state determines which club the manager may choose, recruitment changes that club's next roster, resources and tactics operate on the changed squad, match minutes and injuries settle into the new carryover, and later windows are generated from the resulting roster. Candidate quality and cost are bounded game design values, not real-player evaluation or evidence that a transfer causes match success. This stage runs no model training, formal world-model mechanism study, external provider call, or validation match.

## 28. V3.10 Replayable club finance and wages

V3.10 closes the economic loop around the replayable squad system. Each managed club starts with 8 bounded game credits and owns an append-only finance ledger in the same product-session transaction as the season archive and squad registry. A completed season is settled before the next recruitment window opens; the resulting cash balance, rather than a detached transfer-screen budget, limits what the manager can spend. The fixed window cap remains 8 credits, so the actual allowance is `min(8, max(0, cash balance))`.

```text
validated completed-season archive + historical effective roster
  -> participation revenue = 6
  -> points revenue = floor(league points / 3)
  -> position revenue = league size - final position
  -> objective bonus = 3 when achieved, otherwise 0
  -> wage expense = ceil(sum(player wage tiers) / 6)
  -> replayable settlement delta and balance
  -> cash-limited recruitment allowance
  -> squad transaction + matching finance charge
  -> one atomic product-session write
```

Player wage tiers are deterministic game rules derived from the same bounded quality summary used by squad construction: quality below 0.62 costs tier 1, below 0.70 tier 2, below 0.78 tier 3, and all higher values tier 4. Seasonal wage expense is the ceiling of the roster tier sum divided by 6. If a legacy or synthetic workspace has no usable roster evidence, settlement uses an explicit team-level fallback expense of 6 and labels that weaker evidence source; it never fabricates player-level wages.

Every settlement records the archive identity, roster identity, complete revenue components, wage evidence, balance before, delta, and balance after. Every recruitment charge records the exact squad-transaction identity and negative spend. On workspace load, ledger ordering, identities, arithmetic, bounded balances, duplicate season events, and recruitment-to-squad bindings are replayed. For retained season archives, the historical roster is reconstructed through the preceding squad transactions and the settlement is recomputed from source evidence. Changing a balance, charge, archived result, wage input, or linked squad transaction therefore fails closed.

The active completed season is previewed without mutating persistence, so Studio can show the same projected settlement and cash allowance that next-season creation will commit. Next-season creation performs archive construction, historical-roster reconstruction, settlement, contract validation, squad transaction, and finance charge under one workspace lock and writes once. An unaffordable signing rejects the whole operation; neither a partial archive nor a partial squad or finance entry remains.

Studio presents current cash, next-window allowance, current wage expense, projected post-season balance, and the recent ledger. Candidate options above available cash are disabled, while the backend independently enforces the same rule. The ledger retains up to 256 entries per club; direct settlement source re-verification is limited by the product's retained 12-season archive window, while older entries remain protected by canonical identity and arithmetic replay. Credits, wage tiers, revenues, and candidate costs are fictional bounded game mechanics, not accounting, valuation, salary, or transfer claims about real clubs. This stage runs no model training, formal world-model mechanism study, external provider call, or validation match.

## 29. V3.11 Deterministic living-league ecosystem

V3.11 extends the persistent career world beyond the player-controlled club. When a completed season rolls into the next one, every previous participant receives a settlement through the same finance ledger and wage formula. The manager club alone can receive its frozen objective bonus; non-player clubs receive no invented objective reward. Clubs that remain in the competition then share the same cash-limited fictional recruitment market and the same replayable squad registry used by the player.

```text
completed archived season + historical effective rosters
  -> settle every prior participant in stable team order
  -> intersect prior and next competition membership
  -> reserve the next manager-controlled club for human choice
  -> classify each remaining club from final league position
     top half: selective, at most 1 move, >= 0.025 quality gain, spend <= 5
     bottom half: rebuild, at most 2 moves, >= 0.012 quality gain
  -> enforce post-settlement cash allowance <= 8
  -> replace the weakest same-role player only
  -> append squad transaction and matching finance charge
  -> persist one audited league transition in the same atomic write
```

Candidate generation is unchanged: it is a pure function of the next season index, club identity, prior effective-roster identity, weak formation roles, and fixed cost schedule. The AI policy evaluates all five candidates against the weakest current player in the same role, records candidate and outgoing quality, improvement, cost, threshold eligibility, and selection reason, then ranks eligible upgrades by quality gain, cost, and stable candidate ID. Same-role replacement preserves squad size and positional coverage; the existing squad validator still independently guarantees unique players and at least one goalkeeper.

Control follows the next season rather than the previous one. If the manager changes club, the newly controlled club is excluded from AI recruitment and may use the human recruitment form with its own real post-settlement balance. The former club becomes AI-controlled if it remains in the competition. Clubs entering the competition without a prior result are not given a fabricated rollover decision. A missing roster is recorded as `roster_unavailable`, produces no transaction, and uses the finance layer's explicit team-level wage fallback.

The top-level `league_ecosystem` registry retains up to 12 ordered transitions. Each transition binds the source archive identity, source and target season IDs, continuing participants, next manager-controlled team, every AI decision, settlement identity, and optional squad-transaction identity. On load, retained source archives are joined to the target season, historical rosters are replayed only through the source season, post-settlement cash is recovered from the finance ledger, every AI decision is recomputed, and target-season squad transactions are matched exactly. Altered rankings, thresholds, strategy labels, allowances, selected players, transaction links, or archive evidence fail closed.

Studio shows the latest transition, number of evolving AI clubs, actual move count, player-control boundary, per-club strategy, league position, cash cap, named outgoing and incoming players, quality gain, cost, and explicit no-action reasons. This turns recruitment, finance, standings, and future opponent strength into one visible long-horizon system. These policies are deterministic game rules designed for playability and auditability; they are not learned behavior or claims about real club management. This stage runs no training, formal mechanism study, external provider call, or validation match.

## 30. V3.12 Replayable club identity and coaching evolution

V3.12 gives every club a frozen season identity that is derived after the recruitment window closes and is actually applied to season matches. The snapshot is part of the season state and immutable archive rather than a presentation-only coach label. Its inputs are the target season's effective roster, the prior season's final position, the prior identity when available, and the exact number of target-season recruitment moves.

Six playable identities share the existing tactical-preset engine: balanced, tiki-taka, gegenpress, counter attack, low-block counter, and direct vertical. Each receives a roster-fit score from explicit ability groups. Technical passing and spatial abilities support possession play; pressing, pace, mentality, and power support gegenpressing; pace, vision, shooting, and mentality support transitions; pressing, power, aerial ability, and mentality support a low block; power, aerial ability, shooting, and pace support direct play. The balanced score uses all ten common outfield dimensions.

```text
post-window effective roster
  -> six bounded roster-fit scores
  -> prior league position
     top third: proactive adjustment
     middle third: balanced adjustment
     bottom third: conservative adjustment
  -> prior primary identity inertia = +0.018
  -> stable score/tactic ordering
  -> primary + home + away tactics
  -> frozen Season.club_strategies snapshot
  -> season_manager MatchPlan for every fixture
  -> tactical preset applied in the physics match engine
```

Posture adjustments are small and versioned game rules, not learned coefficients. Proactive clubs add 0.030 to gegenpress, 0.018 to tiki-taka, and 0.010 to direct vertical. Conservative clubs add 0.035 to low-block counter and 0.022 to counter attack. Balanced clubs add 0.020 to balanced play. Prior identity adds only 0.018, allowing continuity without permanently preventing roster or performance changes. Conservative teams choose their away plan between low-block counter and counter attack; other postures use the primary identity home and away.

Every new season profile records the complete style-score vector, risk posture, prior position, prior primary tactic, post-window roster identity, recruitment move count, evidence quality, control boundary, primary tactic, and home/away tactics. A club without roster evidence receives the explicit 0.5 team baseline and is labeled `team_baseline`; no player-level fit is invented. New entrants have no prior position or inherited identity. Changing club changes control, not the club's roster-derived identity.

Fixture resolution has a strict precedence order. AI-versus-AI fixtures use the frozen home and away identity tactics. In a managed fixture, the manager's submitted tactic overrides the manager club identity. The opponent begins from its venue-specific identity; the existing opponent-preparation mechanism overrides it only when prior manager decisions satisfy the frozen observation and dominance gates. Thus `team_identity` now resolves to an auditable season tactic instead of silently falling back to an unrelated native default.

All strategy-enabled season fixtures use `MatchPlan(experience="season_manager")`, including AI-versus-AI matches, so both sides' tactics enter the same physics-official path used by managed tactical gameplay. The match plan stores the season snapshot identity, both profile identities, venue identity tactics, final applied tactics, manager override team, and whether opponent adaptation was applied. Recovery accepts a prior completed report only when this evidence matches exactly.

Workspace loading reconstructs each retained snapshot from the target-season squad registry, target-season transaction counts, source archive, previous snapshot, and matching V3.11 ecosystem transition. For current-season completed match records it also resolves fixture tactics again from the frozen manager decision and opponent-preparation evidence, then compares the recorded strategy object and MatchPlan tactics. Tampering with ability-derived scores, posture, inertia, roster or transition links, applied tactics, profile identities, or report attribution fails closed. Source re-verification follows the same retained-history boundary as the underlying archives.

Studio adds a league-wide identity view and a matchday briefing containing the manager identity, opponent identity, venue baseline, and post-decision applied preview. It states why each profile exists and whether it is roster-derived or a weaker team baseline. These identities are deterministic playability mechanics and auditable simulator interventions, not trained coaching intelligence, win predictions, causal findings, or real-world recommendations. This stage runs no training, formal mechanism study, external provider call, or validation match.

## 31. V3.13 Evidence-bound player development and lifecycle

V3.13 closes the long-term player loop. A completed simulated season now produces one participation-evidence object per club from the persisted match reports. The collector reads only `raw_summary.player_stats[team][player_id].minutes`, binds every accepted report to its match, season and fixture identity, and stores the report SHA-256 together with the exact per-match minutes. Missing reports or missing team statistics are labeled `partial` or `unavailable`; the system never estimates the absent minutes.

```text
completed season fixtures + retained JSON reports
  -> exact per-player minutes and appearances
  -> frozen participation evidence + report hashes
  -> retain source-season effective roster
  -> apply target-season recruitment first
  -> intersect retained player identities
  -> age curve + observed-minute share + frozen sports-science allocation
  -> role-weighted bounded ability changes and one-year aging
  -> append-only development transaction
  -> next effective roster
  -> wages, future recruitment market, lineup, simulation and club identity
```

The age curve is explicit and versioned. Players aged 21 or younger receive a base season delta of `+0.004`; ages 22–24 receive `+0.003`; 25–27 receive `+0.001`; 28–29 receive zero; 30–32 receive `-0.003`; and ages 33 or older receive `-0.006`. Observed participation adds at most `+0.004` for young players, `+0.0015` for prime-age players, and `+0.001` for older players. The manager club's previously frozen sports-science allocation contributes `+0.0005` per point through age 27 and `+0.0004` per point thereafter. AI clubs receive no invented resource bonus. The combined delta is clamped to `[-0.008, +0.012]`.

Each supported ability changes by 1.15 times the season delta when it belongs to the player's position emphasis and by 0.85 otherwise, then remains inside `[0.15, 0.92]`. Retained players with a valid age advance by one year. A player without valid age evidence receives no age or ability mutation. A target-season signing is explicitly recorded as `new_signing` and does not receive retroactive development from a season in which that player did not represent the club. Departed players are recorded but omitted from the result roster.

Ordering is deliberate. Source-season finance uses the historical pre-aging roster. Target-season recruitment also uses that source roster, preserving the market and decision that were visible before rollover. Development then applies to retained players in the post-window target roster. Consequently the new signing remains unchanged, while every later wage calculation, lineup, match, following transfer market, and club-style snapshot reads the developed effective roster. Recruitment and development events are interleaved by target season during replay, preventing either subsystem from silently overwriting the other.

Every transaction binds the source archive, participation evidence, prior target roster and result roster by canonical SHA-256 identity. It stores sports-science points, every player's age phase, observed minutes, appearances, participation share, decomposed delta, per-ability before/after values, departed identities and recomputable summary. Workspace loading first validates and replays the ledger structurally. While the source archive remains inside the 12-season product history, it also reconstructs both source and pre-development target rosters, rechecks locally retained report hashes, rebuilds the transaction from source evidence, and requires exact equality. If an old report file has been pruned, its frozen hash remains in the archive and the source transaction remains reproducible from retained evidence rather than fabricating a replacement.

Beyond the retained source-archive window, each transaction still performs archive-independent integrity checks: season ordering, identities, player ordering, age transitions, bounded numeric values, ability transitions, status consistency, departed identities, observed-minute totals, changed/new/departed counts and mean ability delta are recomputed before roster application. This does not recreate deleted primary reports, so claims remain bounded to the frozen evidence, but it prevents old lifecycle summaries from becoming unaudited mutable labels.

Studio exposes a league-wide lifecycle panel with settled clubs, changed-player count, exact observed minutes, evidence coverage, sports-science allocation, mean ability movement and leading player changes. It labels the evidence boundary directly. These are deterministic fictional development mechanics intended to make multi-season decisions consequential; they are not medical, scouting, valuation, potential, or real-player performance claims. This stage runs no training, formal world-model action-adoption study, external provider call, or validation match.

## 32. V3.14 Replayable contracts, retirement, and academy renewal

V3.14 turns the V3.13 age field into an actual squad-lifecycle constraint. Every club continuing from one simulated season to the next receives a versioned lifecycle transaction after target-season recruitment and before player development. The transaction decrements retained-player contracts, applies bounded renewal decisions, removes contract releases and retirees, and fills every vacancy with a deterministic fictional academy player in the same role. It therefore changes the authoritative effective roster instead of maintaining a separate career-screen roster.

```text
completed source season + source effective roster
  -> deterministic initial contract evidence when legacy players have none
  -> fixed retirement age per club/player identity (36..39)
  -> expiring and retiring preview
  -> manager choice or deterministic AI recommendation (maximum 4 renewals)
  -> target-season recruitment transaction
  -> contract continuation / renewal / release / retirement
  -> same-role academy promotion for every lifecycle exit
  -> target-season development transaction
  -> wages, lineup, substitutions, simulation, next market and club identity
```

Legacy or base-roster players without a career object receive a stable initial term of one to three years from the club and player identity. This initialization is recorded explicitly as `deterministic_initialization`; it is not presented as imported contract data. A continuing contract loses exactly one year. An expiring player selected for renewal receives a three-year fictional term. A newly recruited player receives a three-year term but is not eligible for source-season development. Every academy promotion begins at age 17 on a three-year academy contract.

Retirement is deterministic rather than probabilistic at runtime. Each club/player identity has one frozen retirement age from 36 through 39. A player retires when the age reached after the completed source season meets that threshold. Retirement cannot be overridden by a renewal choice. Players without valid age evidence do not receive a fabricated retirement decision, matching the V3.13 rule that unknown age also blocks development.

The renewal limit is four expiring players per club and season. The manager receives a frozen read-only preview and may select any subset within that limit. The default UI selection uses the same deterministic recommendation as AI clubs: expiring goalkeepers are protected first, followed by higher bounded squad quality and stable player identity. An omitted plan uses that recommendation for backward-compatible automated progression; an explicitly empty plan releases every non-retiring expiring player. The exact manager selection is serialized in `SeasonPlan.manager_retention` and must match the persisted lifecycle transaction.

Recruitment intentionally precedes lifecycle settlement. This preserves the transfer market and outgoing-player choices already shown to the user at the end of the source season. A recruited player receives a new contract and no retroactive source-season growth. A source player transferred out is no longer processed by the old club's contract operation. Lifecycle then precedes development, so released and retired players cannot grow after leaving, while academy replacements are labeled new arrivals and also receive no retroactive minutes. The authoritative replay order for equal target-season IDs is recruitment phase 0, lifecycle phase 1, and development phase 2; events still run when a club has no recruitment transaction.

Academy generation is a pure function of club, target season, vacated role and stable slot. It produces a unique fictional identity, age, role, prospect squad role, composure, fourteen bounded abilities and academy contract. Role emphasis raises only the relevant deterministic ability groups. The validator regenerates the complete academy player independently and requires exact equality, so an old transaction cannot change a youth player's name, position, ability or contract merely by recomputing a result-roster hash. Same-role replacement preserves squad size and goalkeeper coverage.

The append-only `player_lifecycle` registry lives in the atomic product session beside squad, development, finance and ecosystem registries. Each transaction binds the source/target seasons, frozen preview, optional manager plan, prior/result roster identities, contract updates, exits, promotions and recomputable summary. Structural validation independently checks season ordering, hash shape, renewal limit, plan identity, player-set disjointness, every contract state transition, exit reasons, exact academy regeneration and summary arithmetic. While the source season remains retained, workspace loading additionally reconstructs the source roster and the target roster before lifecycle, rebuilds the complete transaction, and requires exact equality. Target `SeasonPlan` control and manager choice are checked separately.

Studio adds a lifecycle preview API and accessible next-season fieldset. Retiring players are read-only; expiring players use labeled checkboxes with live `selected/4` validation and deterministic recommendations preselected. The league-wide history panel reports renewals, contract releases, retirements, academy promotions, named exits, named prospects and whether the decision came from a manager plan or deterministic recommendation. Dynamic values are inserted through DOM `textContent`.

Contract years, renewal capacity, retirement ages, academy identities and abilities are bounded fictional playability rules. They are not contract, employment, medical, retirement, scouting, potential or valuation claims. Released players currently leave the club registry and are not asserted to form a real or simulated global free-agent economy; that would require a separate identity-preserving market registry. This stage runs no training, formal world-model mechanism study, provider call, external service or validation match.

## 33. V3.15 Identity-preserving global free-agent market

V3.15 supplies the separate registry identified by the V3.14 boundary. A contract-released player no longer disappears after leaving one club. The complete frozen player object enters an append-only global pool, keeps the same player identity, ages while outside a club, may retire from the pool, and can later join another simulated club. A player removed to make room for a free-agent signing also enters the pool. Retired players never enter it.

```text
prior global pool
  -> age every player with valid age evidence by one year
  -> remove players reaching their frozen retirement age
  -> expose only entries whose available-from season has arrived
  -> manager gets first optional signing
  -> AI clubs compete in stable team order for remaining candidates
  -> every signing replaces one same-role squad player
  -> ordinary recruitment (phase 0)
  -> global signing (phase 1)
  -> contracts / retirement / academy (phase 2)
  -> player development (phase 3)
  -> ingest signing replacements and contract releases for next window
  -> freeze one global market transition
```

The one-window delay is explicit. A player released while creating season N receives `entered_season_id = season-N` and `available_from_season_id = season-(N+1)`. That player cannot be signed during the same atomic rollover. This prevents a club from releasing a player after its recruitment decision and another club acquiring that player earlier in the same logical window. Existing pool players age before season N's selection; newly entered players do not age again during their entry transition.

The manager may sign at most one free agent per season and receives first priority over the shared window. The frozen `FreeAgentPlan` binds the market identity, incoming player and outgoing player. A signing has zero fictional transfer fee because the player is out of contract, but it is not economically free: the incoming player's deterministic wage tier replaces the outgoing player's tier in every later settlement. The manager cannot use the same outgoing player in ordinary recruitment and free-agent signing, and the backend independently requires that the outgoing player still exists after ordinary recruitment.

Every signing is a same-role replacement. It preserves squad size, player identity uniqueness and goalkeeper coverage. The incoming object is reconstructed exactly from the global entry, then receives the new club identity, a two-year new-signing contract, `gfs_global_free_agent_v1` source, and its origin-club identity. No ability is regenerated and no new fictional person is substituted. The released replacement enters the global pool with its original identity and becomes available one window later.

AI clubs use the same remaining global pool after the manager choice. For each candidate, the policy compares exact simulated registry quality with the weakest current same-role player after ordinary recruitment. An option must improve bounded quality by at least `0.015`. Eligible options are ordered by improvement, incoming identity and outgoing identity; each club signs at most one. Clubs act in stable team order, so an earlier successful signing removes the player before later clubs evaluate their window. Every evaluated option, threshold, selected option, plan and no-action reason is frozen.

The global `player_market` registry contains ordered source/target-season transitions rather than a mutable unproven pool snapshot. Replaying from an empty pool reconstructs aging, pool retirements, signings, new entries and the result-pool identity. Each transition binds the prior pool, complete aging records, manager/AI signings, AI decisions, entries, result identity and recomputable counts. Duplicate global player identities, same-window double signings, premature availability and inconsistent entry seasons fail closed.

Archive-independent validation reconstructs every incoming player exactly from its pool entry, including abilities, age, source, origin and contract. It also recomputes AI eligibility and ranking from the stored evaluated options. While source seasons remain inside the retained history, workspace loading performs the stronger check: it reconstructs every club after ordinary recruitment but before lifecycle/development, reopens the exact prior market, replays manager priority and AI competition sequentially, and requires exact signing, decision and market-entry equality. Lifecycle contract releases must match the corresponding global entries, while signing replacements must match their own entry evidence.

Studio adds a read-only global-market API and an accessible optional signing fieldset. Candidate labels show persistent name, role, simulated age, origin club and exact simulated-registry quality. Selecting a candidate filters outgoing choices to the same role. Live validation rejects incomplete pairs and overlap with ordinary recruitment. The history panel shows pool size, signings, new entries, pool retirements, named cross-club moves, replaced players and the first season each entry can be signed.

This is not a real scouting feed. Candidate quality is labeled `exact_simulated_registry`; the product does not add arbitrary observation noise and call it scouting uncertainty. A future scouting system would need a separately versioned information budget, calibrated error model and hidden ground-truth evaluation. The current pool, contracts, ages, availability and quality are fictional career mechanics, not real employment, medical, retirement, scouting, transfer or valuation evidence. This stage runs no training, formal world-model action-adoption study, external provider call, service or validation match.

## 34. V3.16 Budgeted scouting and hidden candidate truth

V3.16 replaces the V3.15 public exact-quality boundary with a separately versioned information system. The authoritative market still retains complete fictional player objects because simulation, wages, lifecycle settlement and identity-preserving signing require them. Neither the free-agent market API nor the Studio status projection exposes candidate abilities or exact derived quality. The public pool and latest-transition view contain only identity, name, role, age, origin, season availability and safe movement metadata. The manager's own outgoing squad quality remains visible as known club information.

```text
authoritative global player entry (hidden quality)
  -> deterministic baseline observation for every club/candidate/window
     estimate noise amplitude <= 0.035
     nominal half-width = 0.075
  -> manager or AI spends one of two reports
     deterministic scouted observation
     estimate noise amplitude <= 0.012
     nominal half-width = 0.025
  -> choose a same-role replacement from observed estimates
  -> authoritative signing applies the hidden player object
  -> public status emits only a privacy-safe projection
```

An observation is a pure function of market identity, club identity, player identity, hidden simulated quality and information level. It records estimated quality, lower and upper bounds, interval width, level and claim boundary. Bounds are clipped to the simulator's `[0.15, 0.92]` quality domain and expanded when necessary so that the interval contains the simulated truth by construction. This is an explicit software invariant, not an empirical calibration result. A scouted interval is materially narrower than its baseline counterpart, but it still does not reveal the hidden value.

Each club has two unique scouting reports per market window. Reports are append-only in the atomic workspace session and bind market, club, target season, player, candidate-entry SHA-256 identity and deterministic observation. Repeating a report for the same player is idempotent and consumes no second unit. A third unique candidate fails closed. Workspace loading reconstructs the exact historical market entry while its source season is retained, regenerates the observation and requires both entry identity and report equality. Structural validation independently rejects duplicates, malformed intervals, invalid identities and budget excess.

The manager market overlays persisted reports on baseline observations. Candidate labels display estimate, interval and evidence level; the Studio button persists a selected report, preserves the pending replacement choice and displays used/remaining budget. The league-wide scouting panel exposes only recent intervals and aggregate report/window counts. Dynamic content is inserted with `textContent`, and the operation uses the existing CSRF-protected JSON API and file lease.

AI clubs operate under the same two-report budget. They first rank same-role options using baseline estimated improvement, spend reports on the top two stable candidates, recompute those options from scouted estimates and select only an observed improvement of at least `0.015`. The frozen decision contains the budget, reports, evaluated observations, selected option, plan and reason. Independent validation requires every scouted option to have exactly one identical report, no unreported scouted option, exactly `min(2, option_count)` reports, unique option identities and a reproducible threshold/ranking result. Retained-source workspace replay still reconstructs the complete sequential AI market competition exactly.

Truth isolation is tested across both primary response shapes: market previews have no exact `quality`, while the public registry view serializes neither `abilities`, `quality` nor private roster identities, including inside recent transactions. A three-season workspace test verifies narrowing, budget use, idempotence, signing continuity and rejection of a tampered persisted estimate. Additional unit tests reject AI report/option disagreement and scan the serialized public market view for hidden fields.

The coverage contract is deliberately limited. It proves that these generated intervals contain the internal fictional truth because the implementation constructs them that way; it does not establish probabilistic calibration, player-potential validity, transfer value, decision quality or correspondence to real football. No model was trained, no match was run, no formal world-model action-adoption study was authorized, and no external provider or service was called in this stage.

## 35. V3.17 Replayable signing outcomes and scouting accountability

V3.17 closes the missing feedback loop after a free-agent choice. A scouting report is no longer only an input that disappears into the next roster. Once the signed player's target season is complete, the same atomic rollover that archives the season and settles club finance creates one append-only outcome record. It joins the original signing, the information visible at decision time, exact persisted participation, the completed-season archive and the club's already authoritative financial settlement.

```text
free-agent signing in season N
  + baseline or budgeted report visible when selected
  + completed season-N archive
  + exact/partial/unavailable persisted minutes evidence
  + season-N finance settlement and wage evidence
  -> append one free_agent_season_outcome during N -> N+1 rollover
  -> archive-independent structural and arithmetic validation
  -> retained-source exact reconstruction on every workspace load
  -> manager career detail + privacy-safe AI aggregates in Studio
```

The record binds the signing, archive, participation object and settlement with four canonical SHA-256 identities. It freezes the market, club, player, origin, control boundary and observation source. A manager report is recovered from the persistent scouting registry. An AI report is recovered from the exact AI market decision. If the selected option had no paid report, the baseline observation is regenerated from market, club and candidate identity. Report-backed observations must equal the deterministic scouted observation; baseline and report levels cannot be relabeled.

The realized section records arrival quality, absolute estimation error, interval containment, exact observed minutes and appearances, scheduled match count, bounded utilization share, descriptive usage band, participation coverage, player wage tier, team wage expense and the team's final position and points. Arrival quality is reconstructed from the same frozen incoming player object that entered the authoritative roster. The interval must still contain that value. Wage tier uses the existing finance rule rather than a second salary model.

Usage labels are evidence-aware. Complete evidence may produce `core`, `rotation`, `fringe` or `unused` from explicit utilization thresholds. Partial evidence always produces `evidence_partial`, because available minutes are only a lower bound. No available reports produces `evidence_unavailable`. Therefore an absent report can never be silently converted into a claim that a signing was unused. Position and points are contextual facts only; they are never combined into a transfer grade or causal success score.

The registry validator is independent of retained source files. It requires an exact versioned field set, hash shapes, season and market identities, observation/source/control consistency, interval arithmetic, error arithmetic, utilization arithmetic, usage/coverage consistency, bounded wage and standings fields, a fixed non-causal claim boundary and unique season/club/player outcomes. Extra fields such as a causal grade fail closed. While the season remains in the 12-season product history, workspace loading performs a stronger join: it finds the exact market transition and signing by identity, the archived participation object, and the corresponding finance settlement, resolves the original observation, rebuilds the complete outcome and requires exact equality.

Studio exposes detailed outcomes only for signings that were manager-controlled at the time, across the manager's club history. Internal evidence hashes are removed from the public projection. AI signing qualities and errors remain hidden; the manager sees only league-wide counts by evidence coverage and usage category. The panel shows the original estimate, later-known arrival quality, estimation error, minutes, appearances, usage description and wage tier, followed by an explicit warning that usage, points and position do not identify transfer causality.

The new four-season integration test follows a player from contract release through delayed global availability, manager scouting, cross-club signing, completed target season and post-season outcome settlement. It also proves that changing the persisted team-points context causes exact source replay failure. Unit tests cover manager and AI report recovery, wage and utilization arithmetic, partial/unavailable evidence degradation, public AI-truth redaction, internal-hash redaction and rejection of invented causal fields. Extending this test exposed an older floating-point mismatch in multi-season player-development validation; validator and builder now both round each player's mean ability delta to six decimals before aggregation.

These outcomes are decision-accountability evidence inside the fictional simulator, not transfer evaluation, employment advice, player valuation or proof that a signing caused performance. This stage adds no training, executes no product match, authorizes no formal world-model action-adoption study and calls no external provider or service.

## 36. V3.18 Unified sporting-director planning center

V3.18 replaces three adjacent but previously independent next-season controls with one evidence-bound sporting policy. Recruitment, contract retention and the global free-agent market remain distinct authoritative mechanisms, but the manager must now review a single planning brief and freeze one `SportingDirective` before the rollover can execute any of them. The completed-season workflow names this planning review as the next product action, and Studio places the directive in the same next-season form as every constrained choice.

```text
completed source season and effective roster
  + deterministic recruitment market and fictional cash allowance
  + expiring contracts and deterministic retirements
  + global free-agent observations and remaining scouting budget
  + prior manager signing outcomes
  -> independently recomputed role diagnosis
  -> one canonical planning_id
  -> philosophy + risk level + at most three priority roles
  -> validate all recruitment, renewal and free-agent choices together
  -> freeze brief, directive and compliance evaluation in target season
  -> execute the existing atomic rollover
```

The role diagnosis is deliberately small and reproducible. For each supported role it counts current depth, expiring players, retiring players, mean bounded squad quality, mean known age and the number of candidates in both markets. Need is `2 * max(0, 2 - depth) + 1.5 * pending exits + 8 * max(0, 0.64 - mean quality)`, with the empty-goalkeeper case explicitly receiving the full depth penalty. Roles are ranked by descending need and stable role identity; only roles with an available candidate can be recommended. The default philosophy is derived from cash allowance, squad age and pending exits, never from hidden future match results.

The directive has three orthogonal dimensions. Priority roles bind every incoming player across both markets. `win_now` requires each selected incoming estimate or known fictional recruitment quality to improve on the replaced player by at least `0.015`; `youth_pathway` caps incoming age at 24; and `financial_control` caps recruitment spend at the lesser of four credits and the visible allowance, permits at most two renewals and rejects a free agent whose estimated wage tier exceeds the outgoing tier. Balanced and low risk require a paid scouting report for a free-agent signing. Low risk additionally requires the report's lower quality bound to beat the outgoing player's known simulated quality. High risk may act on the wider baseline observation. These are enforceable game rules rather than advisory labels.

The planning brief is a closed, versioned evidence object. Its canonical SHA-256 identity binds the source and target seasons, effective-roster identity, all three market/cycle identities, finance balance and allowance, scouting budget arithmetic, complete outgoing and candidate projections, pending exits, role diagnosis, historical manager scouting-error summary, recommendation, policy contract and claim boundary. Validation rejects extra fields, malformed numeric domains, duplicated identities, unsupported roles, invalid observation intervals, inconsistent budget arithmetic and altered fixed policy text. It then recomputes every role row, recommended role, recommended philosophy and allowance-dependent rule independently, so recomputing the outer hash cannot legitimize an invented diagnosis.

At rollover, validation occurs before any recruitment, signing, contract or roster mutation. A conflict therefore leaves the persisted session byte-for-byte unchanged. The accepted brief, directive and cross-module compliance evaluation are stored in the new season and archive. While the source archive is retained, every workspace load reconstructs the historical roster, deterministic markets, lifecycle preview, finance allowance, scouting overlays and prior outcome scope, rebuilds the brief and requires exact equality. This source replay detects even a structurally valid, self-consistently rehashed alteration. Archive-independent season validation still checks the closed brief and recomputes the compliance evaluation.

The Studio planning center shows role diagnostics, recommendation, philosophy, risk and no more than three priority roles. It reloads when the managed club or a scouting observation changes, preserves the manager's policy selections while adopting the new planning identity, and submits the directive atomically with recruitment, retention and free-agent choices. The backend remains authoritative; direct API clients receive the same conflict rejection. The plan is presented as one coherent sporting-decision window, while the existing ledgers preserve separation of responsibilities and replay order underneath.

This feature provides deterministic fictional planning constraints, not proof that a philosophy, role priority, signing or renewal will improve results. Historical scouting error is descriptive and no causal grade is produced. This stage adds no training, executes no product match, authorizes no formal world-model action-adoption study and calls no external provider or service.

## 37. V3.19 Replayable sporting-strategy feedback loop

V3.19 closes the loop opened by the V3.18 planning center. A frozen directive no longer disappears into an undifferentiated season result. When its target season completes, the atomic rollover creates one append-only `sporting_strategy_review` that joins the original brief and compliance evaluation to the completed archive, exact available participation evidence, final effective roster, manager-controlled incoming players, bounded finance settlement and board review. The review is then projected back into the following season's planning brief as minimal continuity evidence.

```text
season N frozen sporting brief + directive + compliant window execution
  -> completed season-N archive and manager objective result
  -> exact / partial / unavailable player-minute evidence
  -> final effective roster and role depth
  -> bounded finance settlement and board review
  -> append one non-causal sporting strategy review
  -> privacy-minimized Studio review
  -> minimal continuity projection into season N+1 planning
  -> unresolved or completely observed low-usage roles receive +1 need weight
```

The review records execution counts and spend, every manager-controlled incoming identity, acquisition source, role, known fictional arrival quality, minutes, appearances, scheduled matches, utilization share and evidence-aware usage band. It separately records each frozen priority role's source depth, pending exits, incoming count, final depth and final mean bounded quality. Season position, points, objective status, board grade and deltas, revenue, wage expense and finance delta remain distinct descriptive result fields. They are never combined into a transfer grade, philosophy score or causal success metric.

Participation degradation follows the same contract as player development and scouting outcomes. Complete report coverage permits deterministic `core`, `rotation`, `fringe` and `unused` bands. Partial coverage produces only `evidence_partial`; unavailable coverage produces only `evidence_unavailable`. A low-usage continuity signal is legal only with complete evidence. Missing reports can therefore prompt an evidence warning but cannot become a negative player judgment.

Learning signals are a closed deterministic vocabulary: an unfilled priority role, incomplete incoming evidence, completely observed sub-30-percent incoming utilization, a missed independently frozen objective, a negative bounded season settlement, or an explicit no-flag result. These are review prompts, not learned model outputs. Validators recompute signal ordering and content, utilization arithmetic, usage bands, execution totals, wage tiers, priority-role post-depth and mean quality, board domains and finance arithmetic. Unknown fields and invented causal grades fail closed.

The review lives in a separate registry rather than inside the archive or finance evidence. This avoids a circular identity in which the archive hashes a review that itself hashes the archive and settlement. Instead it binds canonical identities for the archive, participation object, settlement, final roster and original planning brief. While the source archive is retained, workspace loading reconstructs the season roster and market signings, locates the exact finance evidence, rebuilds the complete review and requires exact equality. Beyond retention, the closed registry validator preserves structural and arithmetic integrity.

Before rollover, Studio can construct the same review as a read-only projection from the completed active season. After rollover, it reads the persisted review. Both paths produce the same `review_id` and the same next-season planning brief. The feedback projection contains only review identity, season, team, evidence coverage, unresolved priority roles, completely observed low-usage roles, objective status, finance delta and a fixed claim boundary. It excludes the complete squad snapshot, player qualities and internal evidence hashes.

The next role diagnosis marks continuity roles and adds exactly `1.0` to their existing need score. Current squad depth, pending exits, candidate availability and bounded quality remain the primary diagnosis; prior review evidence cannot bypass market availability or any V3.18 policy constraint. Studio labels continuity roles in the unified planning selector and shows the prior review beside current diagnostics. A separate career-history panel shows the full privacy-safe review, incoming utilization, priority delivery and follow-up prompts.

This loop materially improves long-term playability while preserving epistemic boundaries. It helps the manager remember what was planned, what was executed, what was later observed and what remains unresolved. It does not claim that a signing, renewal, role priority or philosophy caused the position, points, finances, board judgment or player usage. This stage adds no training, executes no product match, authorizes no formal world-model action-adoption study and calls no external provider or service.

## 38. V3.20 Deterministic club timeline and consequential situations

V3.20 adds an in-season narrative layer without introducing arbitrary random popups, hidden morale bonuses or untraceable result modifiers. A club situation is derived only from persisted fixtures, completed scores, frozen manager tactics and rotations, the pre-season objective and the current table. Its choice constrains the next `ManagerDecision` through an already implemented tactic or rotation mechanic. The event system therefore creates genuine play decisions while retaining the simulator's existing consequence model.

```text
two or more completed managed fixtures
  -> deterministic trigger audit
     repeated strongest rotation
     midpoint objective pressure
     two-match poor form
     two-match winning momentum
  -> one priority-ordered club situation for the next managed fixture
  -> manager chooses one of two mutually exclusive commitments
  -> backend validates tactic / rotation against the commitment
  -> freeze event, choice, control source and decision in the season timeline
  -> existing lineup, tactic, fatigue and match mechanics execute normally
  -> journal and archive retain the resolved situation
```

The trigger order is explicit. Two consecutive `strongest` rotations first create a squad-load decision: protect the squad by using `balanced` or `rotate`, or push the starters by using `strongest`. At or after the managed-season midpoint, an unmet frozen objective creates an objective-pressure decision: retain the most-used tactic or select a different supported tactic. Before the midpoint, no more than one point across the previous two fixtures creates a form-response decision, while two wins create a momentum-management decision. Both require a clear choice between tactical continuity and a different supported tactic. Stable fixture order resolves every tie.

These commitments do not add a reward or penalty outside the selected match plan. `protect_squad` uses the existing lower-load rotation path; `push_starters` accepts the existing strongest-lineup load. Tactical continuity uses the existing tactic; changing approach requires another tactic already supported by the product. The momentum alternative explicitly avoids claiming that surprise is an empirically valid counter. Score outcomes remain downstream simulated observations, not proof that the event response worked.

Every situation freezes season, club, fixture, matchday, kind, trigger source and exact prior fixture identities, trigger values, two choice contracts, a recommended choice and a canonical SHA-256 identity. A resolution binds the full event, selected choice, tactic, rotation and control boundary. Explicit Studio choices are labeled `manager`. Older API clients that omit the new field remain compatible: the only choice whose constraint matches their submitted tactic and rotation is frozen as `deterministic_compatibility`, so automation is never mislabeled as user agency.

The timeline is part of the atomic season state. Conflicting explicit choices fail before any fixture or file mutation. Replacing an unstarted decision replaces the same fixture's resolution. Season validation reconstructs every historical event from evidence preceding its fixture, verifies event and resolution identities, recomputes the choice constraint against the frozen decision, rejects duplicate fixtures, and requires every decision-side event choice to have exactly one ledger resolution. Removing the timeline while retaining a choice fails closed. A self-consistently rehashed altered trigger still fails source replay.

The matchday journey now includes a club-situation step between briefing and lineup/tactical decision. Studio presents the trigger source, both responses and their tradeoffs in a labeled, live-region-compatible fieldset. Selecting a response synchronizes incompatible tactic or rotation controls before submission; the backend remains authoritative. The manager profile shows resolved count and latest response, including after the season is complete, while the immutable manager journal keeps the full resolution beside the corresponding fixture.

This is a bounded game narrative generated from the simulated season itself. It does not invent player emotions, press quotations, medical claims, supporter beliefs or real club politics. It does not estimate win probability or causally attribute a later score to the selected response. This stage adds no training, executes no product match, authorizes no formal world-model action-adoption study and calls no external provider or service.

## 39. V3.21 Replayable season commitments and board accountability

V3.21 connects the previously separate season objective, frozen club identity, manager match decisions and career review into one multi-week commitment contract. When creating a managed season, the manager now selects one tactical promise and one squad-use promise beside the mandatory board result objective. The choices are visible before the first fixture, retain one season-long deadline and are evaluated only from completed managed decisions. They do not add a second match engine, hidden morale, a win modifier or a retrospective explanation of results.

```text
explicit result objective
  + explicit tactical policy: club_identity | adaptive
  + explicit rotation policy: trust_core | share_load
  + frozen club primary tactic or team_identity fallback
  -> canonical season commitment contract
  -> completed manager decisions become ordered evidence rows
  -> live pending / on_track / at_risk progress
  -> final fulfilled / missed settlement
  -> transparent confidence and reputation component
  -> immutable archive and independent source replay
```

The tactical promise has two exact alternatives. `club_identity` requires the frozen club primary tactic in at least 60 percent of completed managed fixtures. `adaptive` requires at least two distinct tactics and caps the final dominant-tactic share at 80 percent. The squad-use promise is similarly explicit: `trust_core` requires `strongest` rotation in at least 60 percent of managed fixtures, while `share_load` requires `balanced` or `rotate` in at least 50 percent. A season always contains at least three managed fixtures, so the distinct-tactic commitment is achievable. Frequencies describe usage only and never imply tactical effectiveness.

Progress uses completed fixtures rather than a mutable UI counter. Before evidence exists, a policy is `pending`; during the season its current arithmetic is `on_track` or `at_risk`; only a completed season can settle it as `fulfilled` or `missed`. The board objective retains its independently derived `currently_meeting`, `in_progress` and mathematical `unreachable` semantics. Every progress object includes the exact fixture, matchday, tactic and rotation evidence rows, total and remaining managed matches, contract identity, status counts and fixed consequence policy.

Consequences are deliberately bounded and separate from match results. Each fulfilled tactical or squad promise contributes `+3` board confidence and `+1` manager reputation; each missed promise contributes `-4` and `-1`. The existing result-objective, league-finish and decision-coverage components remain separate. The complete component breakdown and per-promise statuses are stored in the board review. A manager can therefore miss a result target but receive credit for honoring declared operating commitments, without the product claiming that those commitments caused any score.

The immutable contract binds season and club identity, managed-match count, selected policies, the frozen identity tactic, every threshold, counted rotation set, board consequence values and claim boundary under canonical SHA-256. Active-season validation reconstructs the complete contract from the raw `SeasonPlan` and club-strategy snapshot. Changing an identity tactic and recomputing the outer hash still fails exact source replay. The final archive stores the same contract, while workspace loading independently rebuilds it from archived plan and club strategy, rebuilds final progress from the manager journal and objective, and then recomputes the board review. Altered promise status, evidence, contract or consequence therefore fails closed.

Studio places the commitment controls inside the existing season form and renders the three promises inside the manager profile rather than opening a detached task screen. The panel announces the evidence count and shows each promise's status, policy and measured share. Its explanatory boundary states that only final fulfillment or breach enters board settlement. The HTTP API parses the same versioned choices and the backend supplies bounded defaults for clients that omit them; direct clients cannot invent a policy or threshold.

This layer makes match-to-match choices answer to a visible longer plan and gives season careers memory, pressure and accountability. It remains a deterministic fictional game policy, not an assessment of real managers, a causal analysis of tactics or evidence that squad rotation changes performance. This stage adds no training, executes no product match, authorizes no formal world-model action-adoption study and calls no external provider or service.

## 40. V3.22 Evidence-bound named-player role promises

V3.22 gives individual squad members a durable place in the manager's season plan without inventing player emotions, private conversations or hidden morale. Before the first managed fixture decision, Studio presents the current effective roster and allows one to three explicit promises. Each promised player receives one unique role: `core`, `rotation` or `development`. A player cannot appear twice and each role can be used at most once, so the mechanic creates a small set of legible responsibilities instead of a bulk optimization screen.

```text
current effective roster after recruitment, lifecycle and development
  -> manager selects up to three unique players and unique roles
  -> freeze player identity, name, position, known age and roster identity
  -> every completed frozen lineup records promised starter / bench status
  -> live start-share progress and season-end manager-accountability settlement
  -> independently collect exact minutes from persisted match reports
  -> complete / partial / unavailable evidence-aware outcome review
  -> board component, archive, source replay and next-season history
```

Role thresholds are fixed game rules. A core promise requires starts in at least 70 percent of known managed lineups and has a descriptive minute target of 60 percent of scheduled match minutes. Rotation uses 35 and 30 percent; development uses 25 and 20 percent. Development eligibility requires explicit age evidence of 23 or younger. Unknown age does not become an inferred youth label. Thresholds are frozen inside the contract and cannot be supplied by a client.

Manager accountability and observed participation are intentionally separate. The board settles only start share, because the submitted lineup is manager-controlled. Every submitted decision also freezes whether each promised player was actually selectable from the matchday squad. Injury, suspension or another authoritative unavailable state removes that fixture from the player's start-share denominator; an all-season absence settles as neutral `excused`, not as a breach or fulfilment reward. Missing availability evidence fails final settlement rather than silently excusing a decision. Each fulfilled named-player promise contributes `+1` confidence and each missed promise `-2`; reputation is unchanged. Actual minutes can be changed by substitutions and match events and can also become unavailable when a report is missing, so they never alter that board component. The outcome object records selection status beside minutes, appearances, minute share and its own observed status rather than collapsing them into one grade.

Participation follows the existing development evidence contract. Complete report coverage permits `fulfilled` or `missed` against the descriptive minute threshold. Partial coverage produces only `evidence_partial`; unavailable coverage produces only `evidence_unavailable`. Missing reports therefore cannot create a negative player judgment. Minute results do not claim that selection caused development, that a player deserved a role, or that the fictional utilization corresponds to real-world ability or potential.

The contract binds season, club, control source, effective-roster identity, total managed fixtures, each player snapshot, exact role thresholds, board consequence values and claim boundary under canonical SHA-256. Explicit Studio plans are labeled `manager`. Older API clients that submit a match decision before the new setup freeze an empty `deterministic_compatibility` contract atomically with that decision, so absence of a promise is preserved without being mislabeled as user agency. Once any contract is frozen it cannot be replaced.

Active season validation checks the closed contract, season/team identity and match count. Workspace loading reconstructs every retained contract from the historical effective roster after the correct recruitment, lifecycle, market and development transactions. A changed player name, age, role, roster identity or threshold still fails when the attacker recomputes the outer contract hash. Final archive validation rebuilds selection progress from the immutable manager lineup journal, rebuilds minute outcomes from the retained participation object, and then recomputes the board review. Altering a status or minute conclusion without changing its sources fails exact replay.

Studio places named-player setup immediately before the first match decision. It lists roster name, position and known age, prevents duplicate players and roles, disables development for ineligible or unknown-age players, and blocks the ordinary decision form until a valid explicit plan is frozen when a playable squad exists. The manager profile then shows starts, known lineups, measured share, threshold and status. The latest completed-season minute outcome is shown beside career evidence with its coverage label. When no playable roster exists, the existing team-level path remains available and is recorded through compatibility control rather than fabricating player evidence.

This stage connects squad building, contracts, player development, match selection and manager career accountability through existing evidence. It does not add a selection-strength bonus, guarantee minutes, infer a player's feelings, create a potential score or causally attribute results or growth to the promise. This stage adds no training, executes no product match, authorizes no formal world-model action-adoption study and calls no external provider or service.

## 41. V3.23 Unified season-first product journey

V3.23 removes a product-level split that remained after the individual season systems had become connected. The top-level Studio workflow previously understood readiness, an active match transaction, a completed season and ordinary single-match history, but it did not map an active season's setup, decision and matchday phases. A valid managed season could therefore be surrounded by correct lower-level controls while the global next-action message still suggested running an unrelated first match or opening a standalone report.

The workflow is now a schema-versioned read model derived from the already validated session and `matchday_command_center`. It does not persist another cursor. Its precedence is fixed:

```text
readiness blocked
  -> active transaction
  -> completed managed season / sporting plan
  -> completed spectator season / final-table review
  -> named-player promise setup when eligible roster evidence exists
  -> manager fixture decision
  -> advance or resume the current matchday
  -> no season: start the persistent season journey
  -> standalone report review
```

The season-first entry is intentional. When no season or match exists, `start_season` is the authoritative next action; `run_standalone_match` remains an explicit alternative for isolated observation or tactical laboratory use. This changes navigation, not simulation semantics. It prevents the single-match research surface from presenting itself as the main complex-product loop while preserving the existing capability.

For an active season the workflow exposes a five-step journey: `season_setup`, `matchday_decision`, `matchday_execution`, `season_review`, and `next_season`. Each step uses only `action_required`, `blocked`, `ready`, `pending`, `complete`, or `available`. The immediate action carries a fixed ID, a fixed local UI target, and only minimal fixture context such as matchday, opponent and fixture identity. Unknown persisted phases fail closed to `inspect_season_state` rather than falling through to an unrelated match action.

Studio renders the same action in the global live status region and provides a named “go to next step” button. Action IDs map to already bound DOM nodes; the client never executes a selector supplied by persisted state. Activation scrolls to the relevant form or control and moves keyboard focus to its first enabled interactive element. A queued or running Web task temporarily suppresses the navigation button and states that the underlying journey will resume after task completion, reducing accidental duplicate submissions while task idempotency remains the authoritative safety boundary.

The journey reuses the player-promise, manager-decision, command-center, board-review and sporting-director contracts. It does not invent new match effects, bypass explicit decisions, infer a successful sporting strategy or turn navigation completion into evidence of user value. Automated source and socket accessibility checks now cover the live region, named navigation control and focus-transfer contract; external assistive-technology and target-user studies remain open evidence gates. This stage runs no training, product match, formal mechanism experiment or external provider call.

## 42. V3.24 Progressive complex-product workspace

V3.24 changes the presentation architecture without shrinking the product. The previous Studio rendered the manager career, isolated matches, paired experiments, fixed-budget studies, world-model evidence, artifact library, recovery controls, release gates and raw JSON as peer sections in one long document. V3.23 supplied one correct next action, but users still had to visually parse unrelated systems around that action.

Studio now exposes four fixed work areas:

```text
manager career
  -> persistent season, matchday, lineup, club and career systems
match laboratory
  -> isolated match, paired comparison and fixed-budget study
evidence center
  -> world-model adoption, artifact library and release gates
operations and recovery
  -> backup, restore and raw runtime evidence
```

The system status, workflow message, safe report links, asynchronous status and remote logout remain globally visible. All other top-level sections declare one fixed `data-workspace-area`. Navigation uses four named buttons with explicit `aria-pressed` state and a separate atomic live description. The navigation is a set of view controls rather than an ARIA tab widget: sections remain ordinary named document regions, so no incomplete tab/tabpanel keyboard contract is implied.

Availability and presentation are deliberately independent. Domain renderers continue to own the HTML `hidden` attribute for conditions such as unsupported research mode or an absent season. Workspace navigation adds or removes only `workspace-area-hidden`. A section therefore cannot become operational merely because a user changes area, and a domain renderer cannot accidentally reveal content from another area.

The workflow maps only fixed action IDs to fixed areas. Before the user chooses a view, `start_season`, promises, decisions, matchday advancement and next-season planning select the manager area; evidence review selects evidence; readiness and invalid-state inspection select operations. A running task preserves the current view. Once a user explicitly chooses an area, the workflow stops moving the page automatically. The choice is stored only in browser `sessionStorage` after validation against the four literal area IDs; it is not product evidence and does not enter the persisted session.

The global “go to next step” remains authoritative even after an explicit view choice. It first reveals the action's fixed owning area, then scrolls to the already bound target and transfers focus to the first enabled control. Persisted text is never interpreted as a CSS selector. Switching views leaves focus on the activating navigation button, preventing focus from being stranded inside newly hidden content.

Code-level accessibility checks cover button names, pressed state, live description, allowlisted progressive disclosure, session preference and reveal-before-focus ordering. A local zero-match smoke run also executed the actual page with a headless browser and confirmed that a readiness-blocked workspace selected operations, hid career and laboratory sections, and retained `resolve_readiness` as the workflow action. This does not replace screen-reader, reflow, zoom, assistive-technology or target-user validation, and it does not award those open gates. No training, match, external provider or formal mechanism experiment is run by this stage.

## 43. V3.25 Authoritative pre-submit manager-decision impact

V3.25 closes the gap between choosing a matchday action and understanding what the product will actually freeze. Previously, the submission path correctly normalized automatic or manual lineups, promised-player availability, in-match rules, club-situation compatibility and opponent preparation, but those consequences appeared only after the decision had already been persisted. The manager therefore had controls without a trustworthy pre-commit model of their combined effect.

The workspace now has one authoritative in-memory normalization operation. Both `preview_manager_decision` and `set_manager_decision` call `_prepare_manager_decision`; there is no parallel preview ruleset. It validates season and team identity, restricts changes to the next unstarted managed fixture, builds or verifies the exact lineup against the effective roster, freezes promised-player availability, validates substitution references, resolves the current club situation and derives opponent preparation. Formal submission then increments the revision and atomically writes the session. Preview instead deep-copies the validated season, applies the same operation to that copy and returns without calling the atomic writer.

```text
manager controls
  -> ManagerDecision strict parser
  -> authoritative in-memory normalization
       -> exact frozen lineup and roster fingerprint
       -> promised-player availability
       -> validated in-match instructions
       -> club-situation choice and compatibility
       -> opponent preparation
  -> preview branch: calculate bounded projections, no session write
  -> submit branch: revision increment, validation, atomic write
```

The preview reports only controlled and deterministic mechanics. Rotation shows the simulator's fixed status penalty and rotation level. The lineup reports its source and exact starter/bench counts. In-match planning reports the number of frozen rules and whether substitutions are controlled. Club situations expose the exact manager or deterministic-compatibility resolution. Opponent preparation is the same derived object that submission will persist.

Season commitments and named-player promises are presented as explicitly hypothetical one-decision projections. The season projection replays only completed fixtures managed by the contract team and appends the proposed tactic and rotation once. The player projection uses a copied fixture list, marks only the proposed fixture completed in that copy, and evaluates the exact frozen lineup and authoritative availability snapshot. Neither projection enters the persisted journal, board review, career record or final evidence. A projected `on_track` or `at_risk` status is therefore decision support, not a settled promise or a claim about match performance.

The protected Web endpoint uses the same strict `ManagerDecision` parser and CSRF boundary as submission. The client uses one payload constructor for preview and submit, debounces control changes and attaches a monotonically increasing request sequence so a slower obsolete response cannot replace a newer preview. Every control change clears the accepted preview revision and disables submission. A successful preview returns the season revision; submission sends it back, and `set_manager_decision` verifies it while holding the same session lease used for the atomic write. A concurrent tab or process that changes the season therefore makes the preview stale and forces refresh instead of silently committing against a different state. Invalid manual lineups or in-match rules are rejected locally and again authoritatively by the backend. Rendering uses DOM construction and text nodes, with an atomic polite live region and busy state.

Tests prove that preview leaves the session bytes and revision unchanged, that the next fixture remains undecided, and that a subsequent submission freezes exactly the normalized decision and opponent-preparation object returned by the preview. Route tests cover CSRF rejection and strict parsing; telemetry recognizes only the fixed route; accessibility audit covers the live region, shared payload constructor, debounce/race contract and non-predictive language. This stage runs no training, product match, formal mechanism experiment or external provider call. It does not estimate score, win probability, causal tactical quality, real-player suitability or real-world performance.

## 44. V3.26 Replayable manager-decision lifecycle ledger

V3.26 connects the product's previously separate decision and evidence surfaces into one match-by-match lifecycle. The simulator already retained the frozen manager decision in the fixture, direct execution facts in the signed report path, the descriptive result in the season, and team/player commitment progress in replayable contracts. Studio, however, exposed only the most recent direct debrief while commitment progress lived elsewhere. A player could inspect each component but could not follow one decision through the whole system.

The new `build_manager_decision_ledger` is a derived read model, not a new mutable event store. It validates the season through `manager_season_profile`, parses every retained `ManagerDecision`, replays team commitments with `commitment_progress_from_evidence`, and replays named-player promises with `player_promise_progress`. A frozen unplayed decision receives a clearly labeled `hypothetical_if_counted` transition. A completed decision receives `completed_evidence`. Neither path changes the fixture, contract, board state or archive.

```text
authoritative frozen fixture decision
  -> decision identity
  -> strict report parser -> direct engine and instruction execution
  -> persisted fixture score -> descriptive result only
  -> prefix replay of season commitments
  -> prefix replay of named-player lineup accountability
  -> canonical entry identity
all entries -> canonical ledger identity
```

Execution evidence is accepted only when the report-derived debrief binds the fixture match ID, manager team, frozen tactic and rotation, direct-runtime evidence grade, descriptive-result boundary and explicit absence of causal attribution. Any mismatch becomes `executed_evidence_unavailable`; it cannot remain a successful execution card. Missing, unreadable, oversized or out-of-window reports also remain unavailable with a reason. This makes a broken evidence link visible without blocking the rest of the season read model.

Report loading is deliberately bounded. The workspace parses full execution evidence for only the most recent eight completed managed fixtures, limiting a status refresh to a fixed number of files and at most 32 MiB per accepted candidate file. Older ledger entries retain the exact frozen decision, opponent preparation, score, commitment transitions, canonical identities and dashboard/report references, but explicitly state that direct execution is outside the live window or unavailable. The complete report remains the drill-down authority.

Long-term accounting is split from the score and from runtime mechanics. Team-level entries show tactical-identity and squad-stewardship status before and after the decision prefix. Player entries show starts, known lineups, excused unavailability, measured start share and threshold. Per-fixture persistent squad-state deltas are not historically retained by the current schema, so the ledger returns `per_fixture_persisted_state_snapshot_not_retained` instead of reconstructing a fictional fatigue or morale change. This missing evidence is an explicit product boundary and a candidate for a future schema version.

Studio renders the six most recent lifecycle entries inside the matchday command center. Each card shows lifecycle state, decision identity, direct mechanics and instruction counts when available, descriptive score, long-term accounting and a safe full-dashboard link. The summary announces direct-execution coverage and a shortened ledger identity. Rendering uses text nodes, named regions and an atomic polite live status; no report content becomes HTML.

Tests cover the frozen/hypothetical state, completed direct-evidence state, deterministic reload identity, match-ID tampering failure, bounded eight-report loading, safe Web rendering and accessibility contracts. The architecture audit verifies that the ledger replays authoritative contracts and does not persist a second state. This stage runs no training, product match, formal mechanism experiment or external provider call. It does not claim that a tactic, lineup, substitution or club choice caused a score, table position, player development or real-world outcome.

## 45. V3.27 Three-phase persistent world-state evidence

V3.27 makes the simulator's cross-match world state part of the manager action
lifecycle. Previously, V3.26 could prove which decision was frozen and which
runtime instructions executed, but correctly reported per-fixture fatigue,
morale and availability changes as unavailable. Reconstructing those changes
from a final carryover file would have mixed later recovery and later matches
into earlier evidence. The season transaction now captures the state at the
three boundaries where it is authoritative.

```text
validated manager decision
  -> capture before_match under the carryover-file lease
  -> atomically reserve the fixture with that snapshot
  -> run and settle the match
  -> capture after_match and persist the replayable match delta
  -> complete every fixture in the matchday
  -> apply matchday recovery once
  -> capture after_recovery and persist the separate recovery delta
  -> archive the complete transition with the season
```

Only fixtures controlled by the career manager receive these snapshots. A
16-team double round-robin therefore retains at most 30 transitions for one
manager rather than copying state for all 240 league fixtures. Each phase is
bounded to 128 players per team, 32 condition dimensions per player, 16 team
dynamics dimensions and fixed-length identities. Roster ability vectors are
not copied. The evidence includes only cross-match carryover fields: fatigue,
morale, media pressure, injury and suspension counters, form, accumulated
discipline and other persisted player-condition deltas.

The first phase can come from a persisted carryover row, a deterministic roster
baseline, or a deterministic team baseline when no roster exists. Source type,
team identity and a canonical source hash travel with every snapshot. Summary
counts are recomputed from the bounded player rows; an unavailable player is
counted once even if multiple simulated restrictions apply. The transition
binds season, fixture, match, home and away identities, all three snapshots,
both derived diffs, recovery completeness and an explicit claim boundary. Its
validator rebuilds the whole transition, so changing a derived delta and merely
recomputing the outer hash is rejected.

Failure and retry preserve the original `before_match` identity. A retry cannot
silently move the baseline forward after a partial failure. Recovery becomes
complete only when its matchday is recorded as recovered, preventing the
season's recovery journal and the fixture evidence from disagreeing. Completed
reports adopted from the older schema remain compatible, but without an
authoritative pre-match capture their persistent-state evidence stays
unavailable. Season history retains and revalidates new transitions rather than
trusting an archived derived object.

The decision ledger presents match and recovery deltas beside the frozen plan,
direct execution, descriptive score and long-term commitments. This closes the
product chain from action to persistent simulated consequence without turning a
state difference into causal attribution. Fatigue, morale and availability are
simulator facts only; injury fields are not medical judgments, and a score does
not demonstrate that the manager decision caused the change or the result.
This stage runs no training, product match, formal mechanism experiment or
external provider call.

## 46. V3.28 Identity-bound world-model manager advisor

V3.28 closes the remaining gap between a capable internal world-model planner
and the player's action-adoption loop. The match engine already constructed
short-horizon tactical policy packets for simulated coaches and cognitive LLM
consumers, but the career manager could neither inspect that comparison nor
record whether it influenced the submitted decision. The product now reuses
that same evaluator through an explicit advisory action; it does not introduce
a second tactical scoring formula.

```text
manager requests advice in research/cognitive mode
  -> verify accepted checkpoint and readiness identity chain
  -> bind next fixture, current season revision and fixed match seed
  -> capture current carryover source identities
  -> construct one deterministic representative pre-match state
  -> compare all seven playable tactics with the existing world-model packet
  -> persist bounded ranking + checkpoint/state identities, increment revision
  -> manager explicitly adopts recommendation or reviews and selects another
  -> authoritative zero-write decision preview
  -> atomic submit rebuilds adoption evidence from advice + final tactic
  -> lifecycle ledger keeps advice, selection, execution, state and result apart
```

The candidate domain exactly matches the product's tactical controls:
`team_identity`, `balanced`, `tiki_taka`, `gegenpress`, `counter_attack`,
`low_block_counter` and `direct_vertical`. Previously the internal pre-match
packet compared named presets but could not represent the club-native vector;
mapping `team_identity` through the generic resolver would have silently turned
it into `balanced`. The evaluator now accepts the exact native tactical vector
and converts it to the same auditable high-level action mixture used by the
other candidates.

Advice construction uses the fixture's eventual season seed and an isolated
`random.Random` stream for SocietyAgent initialization. It therefore does not
consume or reset the process-global random stream. Current squad carryover is
applied, club identity tactics and already-derived opponent preparation enter
the representative state, and the manager's side receives deterministic ball
possession for the comparison. This is still a representative policy proxy,
not the first tick of a fully executed match and not a simulated score.

Only an available packet may be persisted. It must cover every playable tactic
exactly once, contain finite bounded metrics and normalized event probability
distributions, and expose a runtime checkpoint signature identical to the
accepted artifact SHA-256. Candidate order, rank, recommendation, confidence
and top-two margin are replay-checked after canonical hashing. The compact
record retains no latent tensors or full action traces. Carryover source hashes
bind the state used for advice without duplicating roster abilities.

Requesting valid advice is an intentional season mutation: it stores the packet
and advances the revision once. The ordinary decision preview remains zero
write and must refresh against that revision. Any later season mutation makes
the advice ineligible for a new linked adoption. Submission accepts only three
client fields—schema version, advice identity and explicit intent—and rebuilds
the authoritative adoption record from the persisted advice and final tactic.
`adopt_recommendation` is rejected if the tactic differs; a deliberate manual
choice is recorded as `reviewed_then_selected`. Merely loading an old advice
card is not treated as adoption.

Stable mode returns an explicit unavailable boundary without loading the model,
writing the session or pretending that deterministic mechanics are learned
advice. Research readiness failure and model quality-gate closure likewise
produce no advice record. The lifecycle ledger separately renders model advice,
manager selection, explicit interaction, direct execution, persistent-state
change and descriptive score. Agreement is not evidence of decision quality,
and a later result cannot establish that the recommendation caused anything.

Tests cover the complete candidate set, native-vector non-aliasing, checkpoint
identity, event distributions, rehashed ranking tampering, explicit adoption,
manual divergence, stale revision rejection, stable-mode zero writes, CSRF,
strict request fields, telemetry, DOM safety and accessibility. This stage runs
no training, product match, formal action-adoption experiment or external
provider call. Confirmatory recommendation-quality and user-value evidence
remain open gates.

## 47. V3.29 Replayable manager-advisor adoption evidence

V3.28 made the learned model influence an explicit manager decision path. V3.29
makes that influence measurable without adding a second source of season truth.
`decision_ledger.world_model_advisor_summary` is derived only from the existing
fixture advice, explicit adoption record, frozen decision and execution state.
It reports request coverage, explicit adoption versus reviewed manual choice,
descriptive selection alignment, direct-execution coverage, recommendation
diversity, confidence and margin. It deliberately exposes no outcome-effect
estimate.

Every ledger is now replay-validated at construction. Entry identities are
recomputed, lifecycle and advisor summaries are rebuilt from the entries, and
the enclosing ledger identity is checked last. Editing a derived count and
rehashing only the outer object therefore fails semantic replay.

`data/evaluation/manager_advisor_protocol_v1.json` freezes information windows
at 12, 24 and 48 advised decisions. `scripts/manager_advisor_study.py` is a
read-only analyzer unless an output path is explicitly supplied. Before the
first window it reports `insufficient_evidence`; after the count is reached but
before every included decision has executed it reports `window_not_closed`.
Only a closed fixed window can confirm or fail the frozen interaction,
execution, missingness and recommendation-diversity gates.

This protocol evaluates adoption instrumentation, not tactical efficacy. It
runs no matches, performs no training and makes no provider calls. A confirmed
window cannot authorize a score, win-probability, causal, real-football,
product-promotion or academic-promotion claim. Manager-level adoption evidence
and the separate in-match pass/hold mechanism study remain distinct layers.

## 48. V3.30 Optimistic two-phase manager-advice transaction

World-model candidate evaluation is materially slower than an ordinary Studio
state read. The advisor therefore no longer holds the global product-session
lease while loading the checkpoint and evaluating seven tactics. Phase one
briefly locks the session, validates the season and freezes the mode, season
identity, revision, manager, fixture, derived match seed and immutable copies
of the season inputs. Readiness, carryover capture and model inference then run
without the session lease, so status reads and unrelated safe operations are
not blocked by model latency.

Phase two reacquires the lease and performs an optimistic commit. It replays the
current season contract and requires the same mode, season, revision, focus
team, next fixture identity, schedule coordinates, execution state and attempt
count. It also recaptures both teams' carryover source identities and verifies
the configured checkpoint metadata plus the actual checkpoint file SHA-256.
Only then may it persist the advice and advance the revision exactly once.

Any drift returns `advice_inputs_changed_during_generation` with a retryable
boundary and writes nothing. In particular, a manager decision committed while
inference is running remains authoritative and cannot be overwritten by the
older advice request. Stable mode still returns before readiness, state capture
or model loading. Tests prove the inference phase does not own the session
lease and cover concurrent decision, world-state and checkpoint drift.

## 49. V3.31 Confidence-aware manager-advice comparison

Raw candidate scores are not sufficient decision support. The zero-write
manager preview now asks the server to replay a comparison between the model's
recommended tactic and the tactic currently selected in the form. The compact
comparison binds the advice identity, both tactic identities and ranks, plus
recommended-minus-selected deltas for risk-adjusted proxy value, effective
confidence, uncertainty, fatigue cost, structural risk and the complete event
probability distribution. A comparison identity hashes the derived record, and
validation rebuilds it from the original advice rather than trusting client
arithmetic.

Display authority has only two deliberately limited levels. `exploratory_only`
is required when model confidence is below 0.15, adjusted historical trust is
below 0.25, the top-two proxy margin is below 0.005, history is insufficient,
or upstream guidance is low-authority. Otherwise the label is
`bounded_review`; neither level authorizes automatic adoption or represents
validated performance. These are disclosure thresholds, not calibrated
football-effect thresholds.

The accessible preview renders the recommendation and current ranks, all major
proxy deltas and retain/turnover/shot distribution changes. Exploratory advice
changes the action label to “review then adopt exploratory advice.” Adoption
still requires that explicit click followed by the ordinary formal decision
submission. Positive deltas mean recommendation minus current selection; they
are not score, win-probability or causal estimates.

## 50. V3.32 Semantically idempotent manager-advice requests

An advisor request now has retry semantics, not merely optimistic overwrite
protection. Before model inference, Studio validates any advice already bound
to the next fixture and compares its season, fixture, manager, mode, seed,
issued revision, checkpoint artifact and SHA-256, and both carryover source
identities with the current authoritative inputs. It then rechecks the live
session, actual checkpoint file hash and recaptured world state under a short
lease. An exact match returns the persisted advice with response-only
`reused: true`; it performs no model inference, writes no bytes and does not
advance the season revision.

Two identical requests that begin concurrently may both perform inference,
but only one can commit. When the second reaches phase two, it accepts the
first result only if the complete generated advice equals the persisted advice
and all checkpoint and world-state identities still match. It therefore
converges on one advice identity and one revision instead of returning a false
failure or creating duplicate state transitions.

Any changed carryover identity invalidates reuse and causes a fresh bounded
evaluation. A change during either verification phase still returns the
retryable stale-input boundary with zero writes. Tests cover sequential network
retry, concurrent duplicate convergence, world-state invalidation and the
existing manager-decision/checkpoint drift cases. This is an execution and
integrity guarantee only; it adds no recommendation-quality claim.

## 51. V3.33 Runtime-bound advisor execution trace

Manager-advisor evidence now continues beyond request, interaction and frozen
selection into the micro engine itself. At state initialization the engine
captures the complete 22-dimensional tactical base vector that it actually
received. At finalization it records the bounded final vector, the exact set of
changed dimensions and their L1 distance. These are engine inputs and runtime
state, not values inferred from a score or post-match narrative.

The contract distinguishes two legitimate execution mechanisms. A named
tactic uses `locked_preset`, must carry the `studio_user_intervention` source
and must be preset-locked. `team_identity` uses `native_team_vector`, retains
the agent's native archetype, carries `agent_team_identity` and must not claim
that a named preset was locked. Treating these mechanisms as interchangeable
is an identity failure.

The post-match parser requires all tactical dimensions exactly once, finite and
within [0, 1]. It replays changed controls and final L1 distance, verifies the
manager team, side, selected tactic, binding kind and source, and hashes both
the initial vector and the complete binding. Malformed evidence fails closed;
legacy reports remain readable but explicitly lack tactical-binding evidence.

The derived decision ledger then binds advice identity, explicit adoption or
review intent, frozen decision identity and the verified runtime binding into
one execution trace. Its states distinguish awaiting execution,
recommendation executed, reviewed alternative executed, unlinked selection
executed and unavailable evidence. Ledger validation rebuilds every trace, so
editing and rehashing an exported entry cannot convert a pending or alternative
choice into direct recommendation execution.

Studio and archived HTML reports expose the applied vector's pressing, line
height, verticality and other selected controls, plus in-match dimensional
change. The trace deliberately contains no outcome-effect estimate and never
authorizes a score, win-probability, causal-effect or real-football claim.

## 52. V3.34 Zero-training four-arm adoption smoke

The action-adoption runner now separates code-path verification from the
sealed 24-run mechanism study. The smoke command executes one shared fixture
and seed under four isolated arms: the M0 no-advisor physics baseline, the C1
deterministic rule-fallback baseline, the sealed M1 checkpoint with planning
disabled, and the same checkpoint with direct action policy enabled. Each arm
uses only 180 simulated seconds, runs no training, makes no provider call and
writes a dedicated diagnostic artifact without creating or changing formal
progress.

The formal runner remains authorization-gated, resumable only under identical
protocol, checkpoint and code identities, and fixed at 24 runs. When that
budget is complete, analysis additionally exports all flat arm rows to CSV and
a reviewer-readable Markdown gate summary. Both remain mechanism evidence:
neither the smoke nor the formal export authorizes product, academic, causal
outcome or real-football claims.

## 53. V3.35 Unified manager product journey

Studio now presents the complex career system as one state-derived lifecycle:
club planning, pre-match decision, match execution, post-match evidence and
long-term operations. The rail is a projection of the existing canonical
season and matchday command-center state; it does not persist a second
workflow. Its active state is therefore consistent with the existing
next-action controller, task recovery and fixture identities.

World-model advice is no longer visually isolated from the product loop. It is
requested inside pre-match decision, explicitly adopted or rejected before the
decision freezes, verified against the runtime tactical vector during match
execution, and exposed again through post-match evidence and the long-term
decision ledger. Research lab, evidence and recovery remain separate
workspaces, but the manager career itself is one continuous product journey.

## 54. V3.36 Confirmed action adoption and full-match outcome gate

The sealed action-adoption mechanism study completed its full 24-run budget
without exclusions, training or provider calls. The prediction-only negative
control retained zero influenced opportunities. The direct-action arm reached
1954 opportunities, influenced 1951, produced 25.246 expected and 25 realized
shared-random-number action changes, and changed all 12 matched pairs. Every
frozen mechanism gate passed. A separate verifier replays the decision from
the exact protocol, checkpoint, code identity and progress rows, then verifies
the CSV and Markdown export hashes.

This result closes the structural question of whether the validated learned
signal reaches the normal sampler and changes micro-actions. It does not close
the downstream outcome question. A new full-match protocol therefore binds
the confirmed mechanism decision itself, the same checkpoint and all
decision-relevant engine/statistical files. It freezes 30 matched fixture-seed
pairs for both M0 and action-policy M1, 5400 simulated seconds per run, one
fixture-stratified paired bootstrap primary comparison and descriptive-only
secondary behavior metrics. No interim effect is exposed and optional stopping
remains forbidden.

Evidence Center reads both layers separately. It shows the completed mechanism
budget and realized changes, then the independent full-match run count and
decision state. The interface states that mechanism confirmation is not match
improvement, real-football causality or product/academic promotion.

The full-match gate subsequently completed all 60 runs: 30 M0 and 30 M1 over
the exact preregistered fixture-seed pairs. M1 changed at least one observable
in every pair, but its external-calibration-loss delta was -0.031893 with a
95% fixture-stratified paired interval of [-26.061955, 5.137430]. The
external-validity gate and the upper-bound-below--0.10 gate failed. The frozen
decision is `inconclusive_keep_research_only`; stable authority remains with
M0 and no product or academic promotion is supported.

`scripts/verify_action_outcome_result.py` rebuilds the 10,000-draw primary
analysis from the completed progress rows, verifies the exact 30 paired units,
protocol, checkpoint, mechanism prerequisite and critical-code identities,
and compares the replayed decision field-for-field. Evidence Center exposes
the interval, 30/30 behavior-changing pairs and failed promotion state without
turning descriptive shot, goal or possession differences into efficacy claims.

## 55. V3.37 User-facing world-model causal fork

Research Lab now turns the previously separate action-policy switch, match
runner, replay and evidence views into one recoverable product operation. A
`world_model_fork` task reserves one pair transaction and atomically executes
two worlds: a prediction-only baseline with `MATCH_WM_PLAN=0`, followed by a
quality-gated action-policy treatment with `MATCH_WM_PLAN=1`. The fixture,
home and away tactics, model checkpoint identity, fast/full configuration and
explicit seed are held fixed. The action-adoption policy is the only permitted
intervention; tactic drift or the wrong policy transition makes the pair
ineligible and the comparison builder rejects it.

The pair uses the existing durable task and match ledgers instead of adding a
second simulation path. Idempotency prevents duplicate submissions, the pair
transaction preserves baseline/treatment identities across process recovery,
and the treatment reuses the baseline random seed. Completion exposes three
safe artifacts through Evidence Library: the baseline report, treatment
report and synchronized comparison dashboard. Each individual match report
states its policy role, while the comparison reports eligibility, intervention
metadata, bounded metric deltas and clock-aligned replays without claiming
cross-world event correspondence.

This workflow improves intervention visibility, not the sealed research
decision. One fork supports only simulator-local attribution for its exact
fixture, tactics, checkpoint and seed after all structural checks pass. Its
score or metric deltas do not establish population effects, outcome benefit,
real-football causality, product promotion or academic confirmation. M0 remains
the stable authority and M1 remains research-only. This architecture change
requires neither model training nor an external LLM/provider call.

## 56. V3.38 Identity-bound policy propagation explorer

The causal-fork comparison now stores a bounded `policy_propagation` object
instead of asking the interface to infer a story from final score deltas. It
joins a realized action-adoption record to treatment replay only by the exact
runtime `opportunity_id`, team and timestamp. Duplicate identities, malformed
records, a missing replay or a failed pair-eligibility check revoke local
attribution rather than being repaired by time proximity. At most 40 valid
changed decisions are retained for presentation, while the summary preserves
the complete valid count and reports truncation, malformed records and
duplicate identities explicitly.

For every retained decision, the explorer displays the counterfactual
baseline action, sampled treatment action and direct-execution state. When
both replays are available it also compares passes, shots, goals, turnovers
and total actions in fixed 30-second and 120-second ranges beginning at the
decision clock. It counts any additional policy changes inside each range.
These windows align time only: they never match events across diverged worlds,
and `downstream_causal_attribution_authorized` is therefore always false.

The persistent task stores only a bounded propagation digest. Evidence
Library shows changed, directly observed and locally attributable counts and
links to the full comparison artifact; it does not copy decision records into
the general Studio status response. Legacy or damaged comparisons degrade to
an explicit unavailable state. The product can therefore show where the
world model visibly entered play without converting downstream descriptive
movement into an unsupported causal chain or changing the sealed M0/M1 gate.

## 57. V3.39 Selectable in-match deterministic branch anchor

The world-model fork is no longer restricted to a whole-match policy contrast.
The user selects a match minute. Both worlds replay the same fixture, tactics,
checkpoint and seed with action authority disabled before that point. The
treatment enables the quality-gated policy only when the simulated clock
reaches the selected time; the baseline remains prediction-only throughout.
All four planner entry points receive the live match state and enforce the
same delayed-activation boundary.

Immediately before actions execute on the first eligible tick, the micro
runner creates a canonical SHA-256 branch identity. Its scope includes the
observable match state, NumPy generator state, scheduled-event cursor,
passing/shot/aerial statistics, player statistics and substitutions,
continuous/subtick diagnostics, and manager-rule runtimes. Each individual
report stores this anchor. A paired comparison with a declared branch time is
eligible for simulator-local policy attribution only if both anchors are
present, refer to the same requested time and have the exact same identity.
Missing or modified evidence fails closed as
same_pre_intervention_branch_anchor=false.

This is intentionally a deterministic replay anchor, not a serialized Python
process checkpoint. The current monolithic engine also owns mutable state in
sub-engines, queues and cognitive runtimes, so pretending that
MatchAffectiveState alone can be resumed would create a false checkpoint.
The product and comparison artifact expose deterministic_replay_only as the
resume capability. A future process-level snapshot may optimize replay, but it
must first serialize and restore every decision-relevant mutable component and
prove equivalence against this identity-bound replay contract.

The Web workflow provides a 0--90 minute selector (45 minutes by default),
persists seconds in WorldModelForkPlan, and includes branch verification in
the bounded task/evidence digest. This stage changes code and product
evidence only: it does not train a model, execute the sealed formal protocol,
call an external provider, or revise the existing research-only M1 decision.

## 58. V3.40 Versioned authoritative product clock

The in-match fork audit exposed a pre-existing clock ownership ambiguity. The
micro runner assigned the end of a tick before calling the affective engine,
while the affective engine also advanced state.clock_seconds by dt. The
result was a one-tick offset for actions, cooldowns, world-model observations
and logs whenever affective dynamics were active; disabled affective dynamics
followed a different clock path.

Studio now explicitly sets MATCH_AUTHORITATIVE_TICK_CLOCK=1. Under the
authoritative_tick_v2 contract, the runner presents the start of the interval
to integration, then normalizes the state to the exact interval end before
manager rules, action sampling, adoption records, cognitive processing and
world-model completion. Thus a 30-second fast run exposes action boundaries at
30, 60, 90 and 120 seconds, and a policy scheduled for 60 seconds is disabled
at 30 and enabled at 60. Each match report stores both the logical duration and
final state clock. Studio integrity fails closed if this contract is missing.

The legacy path remains legacy_affective_offset_v1 when the explicit product
environment flag is absent. This is deliberate compatibility, not endorsement:
the completed frozen studies did not bind the micro runner in their historical
code-identity list, so globally changing their runtime semantics or rewriting
their stored hashes would be scientifically misleading. New Studio product
runs use V2, while prior sealed results retain their original interpretation
and research-only decision. A future confirmatory protocol must bind the clock
contract and runner identity before collecting new evidence.

The branch hash includes the clock contract, and paired attribution additionally
requires both reports to declare authoritative_tick_v2. Single-match reports,
the comparison dashboard and Evidence Library show the branch minute and clock
or prefix-identity evidence. This stage executes no training, formal study or
external provider call.

## 59. V3.41 Evidence-graded counterfactual future product

The world-model fork previously exposed several correct but separate artifacts:
pair eligibility, action adoption records, synchronized replay, downstream
windows and a flat metric table. A user still had to reconstruct the meaning of
the experiment manually. The comparison contract now emits one versioned
counterfactual_future_summary that binds the intervention point, baseline and
treatment identities, complex-system metric layers and claim authority.

The summary presents one intervention and two futures. Its evidence ladder is
strictly ordered:

1. shared_prefix verifies the identity-bound pre-intervention state and the
   authoritative_tick_v2 clock;
2. policy_to_action counts realized, directly observed and locally
   attributable sampled-action changes;
3. action_to_trajectory reports whether bounded replay windows exist but
   always denies downstream causal attribution;
4. trajectory_to_outcome classifies measured differences as descriptive,
   absent or incomplete and never promotes them to match-outcome causality.

Metrics are grouped into score, chance creation, possession/progression,
complex-system state and world-model mechanism layers. Missing values remain
missing rather than becoming zero. The result distinguishes an ineligible
comparison, no realized action divergence, an unverified action divergence, a
locally attributable action with descriptive future differences, and a locally
attributable action without measured downstream differences.

The comparison dashboard now leads with this unified view before detailed
propagation and replay. Background tasks retain a bounded digest, and the
Evidence Library displays the same state while hard-coding match-outcome and
real-football causality to false. This is still a single fixture/seed simulator
fork, not a population estimate, real-football causal result or model-promotion
gate. This stage runs no training, formal experiment, match study or external
provider call.

## 60. V3.42 Fixed-budget multi-timepoint future set

The product previously stopped at one branch time and one pair of futures.
That made local evidence inspectable, but it did not let a user see whether
world-model action adoption was sensitive to when it entered the same match.
Studio now composes the existing identity-bound fork into a fixed set of two
to four increasing, unique branch times. The default set is minute 30, 45 and
60. Fixture, seed, both tactics, fast configuration, checkpoint, baseline
policy and treatment policy remain fixed across every scenario.

This feature deliberately does not modify the frozen action-adoption engine,
planner, maximum 0.35 probability blend or any sealed protocol identity. Each
scenario is still the same predict-only versus action-policy contrast and must
pass the existing authoritative-clock, shared-prefix and comparison checks.
The additional dimension is branch time only.

WorldModelForkSetPlan freezes the complete scenario budget before execution.
The persistent worker writes a protocol and a fail-closed progress ledger,
uses stable transaction identities per branch, and resumes only a valid prefix
of the registered times. Interim analysis is null and interim ranking is never
disclosed. Artifact paths must resolve inside the current Studio match output;
seed, policy assignment, both fixed tactics, branch order and eligibility are
revalidated before aggregation. Completed results also reject modified claim
authority, branch identity or any attempt to authorize an outcome claim.

The final report counts eligible anchors, realized action-divergence scenarios,
locally attributable actions and descriptive future differences. It may state
that the measured signatures vary across the registered times. It does not
rank times, select a best intervention time, perform significance testing or
authorize match-outcome causality, population inference, real-football
causality or model promotion.

Studio exposes this as one coherent future workbench rather than another
simulation engine. The form preregisters branch minutes, the background task
shows completed scenarios without effects or ranking, and the Evidence Library
opens one aggregate dashboard with links back to every two-world comparison.
The original single-fork API remains available for compatibility and focused
debugging. This stage changes code and product contracts only; it executes no
training, formal protocol, product match study or external provider call.

## 61. V3.43 Identity-bound manager prematch future bridge

The fixed future set previously lived only in the research Lab. It could
demonstrate simulator-local action and trajectory divergence, but it was not
connected to the manager's actual next fixture. The career product now offers
an optional bridge after the manager decision is frozen. It derives the exact
official fixture, matchday seed, fast configuration and both tactics from the
authoritative persisted season. When club-strategy snapshots are present it
uses the same resolver as matchday execution; otherwise it uses the frozen
manager decision and deterministic opponent preparation.

One canonical context identity binds the season id and revision, fixture
identity, manager decision, opponent preparation, optional club-strategy
resolution and all execution controls. The queue independently validates the
shape, hashes and equality with the future-set plan. The worker then rebuilds
the context from the persisted season immediately before execution and again
after all registered branch times finish. A changed revision, fixture,
decision, preparation or strategy fails the task without publishing a result.
The executor performs a final defense-in-depth control check and stores the
context in its frozen protocol, result and dashboard.

There is no second persisted season state. The career view derives queued,
running, interrupted and completed future sets from the existing task queue,
and marks evidence current only when its revision and next-fixture identity
still match the active season. Historical artifacts remain inspectable but
are not silently reused as current advice. The official matchday remains the
only authority that advances the season.

This bridge makes the research capability perceptible in the product:
freeze one manager decision, inspect several preregistered entry times, and
return the resulting two-world evidence to the same career surface. It does
not select a best time, predict a score, estimate a population effect, prove
match-outcome causality or provide real-football coaching advice. This stage
changes code, tests and zero-execution documentation only; it runs no model
training, formal experiment, match study or external provider call.

## 62. V3.44 Manager future-review decision loop

The manager bridge previously returned future-set evidence to the career
screen, but viewing it did not become part of the authoritative decision
lifecycle. The manager could change a tactic afterward, yet the product could
not distinguish an ordinary edit from an explicit evidence review. This left
the world model visible but still disconnected from the final user action.

Studio now records one replayable review receipt when the manager explicitly
chooses either keep_after_review or revise_after_review. The source must be a
completed manager-bound future-set task whose full context still matches the
current season revision and next unstarted fixture. A keep receipt is valid
only when the final normalized decision identity is unchanged. A revise
receipt is valid only when it changes. The receipt binds the task, source
context, original and final decision identities, registered branch times and
bounded mechanism counts. Best-time recommendation, score prediction,
outcome-effect estimation, population inference, real-football causality and
promotion authority remain false.

The update occurs under the existing season lease. Decision normalization,
lineup validation, club-situation resolution and opponent preparation reuse
the official manager-decision path. The receipt is appended to the fixture,
the season revision advances atomically, and the previous future set becomes
historical. A revised decision may be explored again, producing a new context
and another receipt; duplicate task or context identities and histories above
the fixed per-fixture cap fail closed.

Review receipts flow into the existing manager season profile, decision ledger
and archived season journal. The ledger reports only explicit keep/revise
counts and whether the reviewed simulator sets contained action divergence or
timing sensitivity. It never compares those interactions with scores as an
effect estimate. The task queue remains execution evidence, the persisted
season remains the sole career authority, and official matchday execution
remains the only operation that advances competition state.

This stage completes a product interaction loop, not an efficacy result. It
changes code, tests and zero-execution documentation only and runs no model
training, formal experiment, match study, human study or provider call.

## 63. V3.45 Replayable per-timepoint manager evidence

The completed future-set task previously exposed only aggregate counts to the
manager screen. Its official artifact already contained one row for every
preregistered branch time, but those rows remained behind a separate research
dashboard. This preserved scientific caution while leaving the product
experience fragmented: a manager could record a review without retaining the
specific mechanism chain that was visible at review time.

Every newly completed future-set task now carries a bounded public projection
for its two to four frozen branch times. Each row records the branch minute,
eligibility and prefix-anchor state, realized action changes, locally
attributable changes, descriptive future differences and explicit false
outcome/real-football causal authority. A canonical scenario identity binds
the complete row. Validation replays each identity, enforces status semantics
and rebuilds the aggregate counts, status distribution and timing-sensitivity
flag from the rows. A task or receipt whose row facts and aggregate disagree
fails closed.

Version 2 manager review receipts preserve the validated rows inside the
existing fixture review history. Version 1 receipts remain replayable for
backward compatibility, while all newly executed tasks produce the richer
contract. Studio renders the same rows inline on the current manager future
set and again from the immutable receipt in the decision ledger, so task
pruning does not erase the evidence behind a recorded decision. No second
season state, preferred branch, best-time ranking or automatic decision is
introduced.

This is a product observability and evidence-continuity improvement. It does
not add training, execute matches or formal studies, estimate decision quality,
or upgrade simulator differences into match-outcome or real-football causality.

## 64. V3.46 Concrete world-model action mechanism cards

Per-timepoint evidence made the future set visible inside the manager journey,
but a count such as one changed action was still abstract. The official paired
comparison already retained the concrete policy-propagation record, including
the team, authoritative match clock, baseline sampled action, action-policy
sampled action, model recommendation, direct runtime observation and local
attribution eligibility. It also retained same-clock 30-second and 120-second
event-count windows.

Newly generated scenario evidence is version 2 and carries at most the first
three changed-action examples per branch. Opportunity identifiers are hashed
instead of exposed. Each example and each downstream window has its own
canonical identity. Windows contain bounded deltas for actions, passes, shots,
goals and turnovers plus the number of other policy changes in the window.
They always set causal authority to false. Truncation is explicit and must
agree with the total changed-action count.

Manager review receipt version 3 requires version 2 scenarios. Existing v1
aggregate-only and v2 per-timepoint receipts remain valid, and mixed versions
fail closed. The current manager future card and archived decision ledger use
safe DOM text nodes to show action transitions such as hold to pass and the
bounded descriptive windows. The ledger summary counts retained examples,
locally attributable examples and examples with descriptive windows.

This closes another product-explanation gap without selecting an action,
ranking a branch time or claiming a downstream causal path. Direct runtime
identity may authorize only the local sampled-action change; every later
trajectory and outcome difference remains descriptive. No training, match,
formal study, human study or provider call is executed by this change.

## 65. V3.47 Future review to official runtime closure

The manager ledger previously retained the reviewed future scenarios and the
official postmatch tactical binding, but it did not derive one explicit chain
between them. A user could see both pieces without knowing whether the
terminal review still matched the final frozen decision, whether another
review followed it, or whether an ordinary unreviewed edit superseded it.

The ledger now derives a replayable execution trace from the authoritative
fixture only. Each review is classified as followed by a later review, selected
for the fixture, or superseded by an unreviewed edit. The terminal review's
final normalized decision identity must equal the fixture's final decision
identity before it can enter the execution chain. Pending fixtures remain
explicitly awaiting execution. For completed fixtures, verification requires
direct postmatch evidence whose tactical binding reports the same applied
tactic as the final selection. Missing or mismatched direct evidence remains
unavailable rather than being inferred from the score.

The trace and its aggregate summary are canonical derived evidence. Ledger
validation rebuilds both and rejects even rehashed tampering. The manager
screen renders the review sequence, terminal-selection state and verified
engine tactic inline with the existing decision journal. It also states the
boundary at the point of use: the chain does not match a prematch simulated
path to the observed score, evaluate forecast accuracy, estimate an outcome
effect or authorize causality.

This completes an instrumentation and product-understanding loop:
world-model evidence review, explicit user decision, frozen decision identity
and official runtime tactical adoption. It does not establish that the world
model improved the decision or the result. The change adds code, tests,
interface evidence and architecture guards only; it executes no training,
match study, formal protocol, human study or external provider call.

## 66. V3.48 Official-match world-model action execution

Research and cognitive manager matches already ran the quality-gated
world-model action policy. Their full match reports retained the probability
adjustments, shared sampling uniform, baseline counterfactual action, sampled
action and exact ball-event identity links. However, those facts were visible
only in the standalone match report and latest-match research panel. The
manager postmatch debrief and historical decision ledger did not consume them,
so the official career surface still could not answer whether the world model
changed an action during the manager's match.

A new bounded projection reads the authoritative match report and keeps only
the manager team's retained records. It validates aggregate hierarchy,
truncation semantics, unique opportunity identities, action vocabulary,
shared-uniform changed-action semantics and all replay-link counters. Raw
opportunity identifiers are replaced by match-and-team-bound hashes. At most
five examples are retained, prioritized by realized action change, nonzero
influence and probability distance.

The projection separates three evidence levels:

1. the action policy changed a sampling probability;
2. under the same sampling uniform, the adjusted policy selected a different
   action, permitting simulator-local action attribution;
3. for pass or shot actions, the decision record matched one exact logged ball
   event by opportunity identity, action, team and timestamp.

Hold remains a direct decision record without a ball trajectory by design.
Cross has a distinct replayable trajectory from V3.64 onward. Missing or
ambiguous event links are counted explicitly. If source
records were truncated, manager-team record coverage is marked incomplete
rather than inferred. Invalid action evidence fails closed inside its own
subview while valid tactical and instruction execution evidence remains
available.

The postmatch debrief now carries this projection. The decision ledger
validates its canonical identity, derives cross-fixture counts and shows the
same bounded examples in the current matchday and historical journal. This is
still not an outcome result: no score comparison, tactic-quality estimate,
trajectory attribution, population inference or real-football causal claim is
authorized.

This stage connects already existing official runtime behavior to the unified
manager product. It adds no second state and executes no training, match,
formal experiment, human study or external provider call.

## 67. V3.49 Unified manager world-evolution thread

The manager product had accumulated the correct evidence components but still
presented them as separate cards: prematch future review, frozen decision,
tactical runtime binding, official world-model action records, observed match
result and persistent world-state transition. A user had to infer the ordering
and, more dangerously, could mistake co-occurrence for a causal edge.

Every managed-fixture ledger entry now derives one six-stage world-evolution
thread from those existing authoritative components. Each stage and each of
the five connecting links has a canonical identity. Ledger validation rebuilds
the complete thread, including link status, from the source fields and rejects
even a rehashed edited thread. No second journal or mutable world state is
introduced.

The links deliberately have different meanings:

1. future review to frozen decision is an identity binding or an explicit
   continuity break;
2. frozen decision to tactical runtime is an official engine binding;
3. tactical runtime and world-model actions share an official match context,
   with no asserted causal direction between them;
4. world-model actions link only to locally attributed sampled-action changes
   and exact retained ball-event identities;
5. the observed match links to the persisted simulator state transition, not
   to a claim that the manager decision or action policy caused the score.

Pending fixtures, stable-mode non-applicability, legacy reports, missing action
records and missing persistent state remain distinct states. The thread records
continuity gaps instead of silently skipping them. It reports both an official
runtime-chain completeness flag and a stricter world-model runtime-chain flag;
neither is an efficacy metric.

The current matchday debrief and historical decision journal render the same
ordered timeline. Users can now follow one world from evidence review through
execution and into persistent fatigue, morale, injury and suspension state
without navigating independent research panels.

This stage reduces product fragmentation and clarifies the complex-system
experience. It does not execute training, matches, formal experiments, human
studies or provider calls, and it does not estimate score improvement or
real-football causality.

## 68. V3.50 Canonical manager counterfactual intervention workspace

The prematch manager product already supported frozen decisions, fixed-budget
future sets, per-timepoint mechanisms and explicit keep-or-revise receipts.
Those capabilities were scientifically bounded but the browser still inferred
their workflow from several task cards. Task state, current fixture binding,
available evidence and the next manager action therefore had no single
replayable product contract.

Studio now derives one manager intervention workspace from the current season
view and the already validated future-set projections. It is a five-stage
state machine:

1. freeze the intervention decision;
2. generate bounded, preregistered futures;
3. inspect simulator-local action mechanisms;
4. record an explicit manager review;
5. advance the official football world through the existing matchday action.

The workspace normalizes the frozen decision, recomputes its identity, checks
future-set fixture, revision and decision binding, retains only bounded
scenario identities and aggregate mechanism counts, and links a review only
through its persisted receipt identity. Every stage and the full workspace
have canonical identities. Validation rebuilds the complete object and rejects
even a rehashed edited workflow.

Queued, running, interrupted, completed-with-scenario-evidence,
completed-with-aggregate-only, review-ready and reviewed-and-refrozen states
remain distinct. Projection drift, invalid identities and missing completed
scenario evidence are visible continuity gaps. An interrupted current future
set exposes the existing safe-resume operation directly in the manager panel.

The same workspace identity is published in the season view and matchday
command center. The browser renders its five ordered stages, evidence totals
and one derived next action before the detailed per-set cards. It no longer
has to infer the overall journey from independent panels. Existing future-set
artifacts and review receipts remain the authorities; the workspace is
read-only and introduces no second season state.

This is a workflow-cohesion and product-comprehension improvement. It does not
rank intervention times, select a tactic, execute a match, predict a score,
estimate an outcome effect, authorize real-football causality, run training or
call an external provider.

## 69. V3.51 Season-level manager world navigator

The canonical intervention workspace unified the prematch workflow, while the
world-evolution thread unified each completed fixture. They still appeared in
different parts of the manager surface. The product could explain one current
intervention and one historical match, but it had no canonical answer to three
basic navigation questions: where the manager is now, what the next action is,
and which completed simulator worlds can be reopened.

Studio now derives one season-level manager world navigator after both source
objects have passed their own replay validation. Its current chapter references
the exact intervention-workspace identity, current fixture, frozen decision,
five workflow stages, bounded evidence summary and continuity gaps. Exactly one
primary action is selected from the workflow state: freeze, generate, wait,
resume, retry, review or advance the official world. Optional exploration and
official advancement remain explicit secondary actions rather than being
silently selected.

Every executed manager-ledger entry becomes an immutable historical world
chapter. A chapter binds the ledger-entry identity, complete world-thread
identity, all six stage identities, descriptive score, simulator-local action
change count, persistent state-transition identity and declared continuity
gaps. Pending execution is represented only by the current chapter and is not
duplicated in history. History is recent-first and bounded to 64 visible
chapters while preserving the authoritative completed count and an explicit
truncation flag.

The navigator verifies the content hashes of its intervention workspace,
ledger entries, world threads and individual stages before projection. Its own
identity is then replayed from those sources; rehashing an edited navigation
result cannot authorize it. The season view and matchday command center expose
the same navigator identity.

The Web command center now begins with this single navigation entry. It shows
the current location, ordered active stages, one actionable next-step control
and recent completed worlds. The control scrolls and focuses the existing
authoritative form, future-set action, review action, recovery action or
matchday button; it creates no new mutation route. Detailed prematch evidence
and postmatch ledgers remain progressive-disclosure layers underneath.

This change joins the product experience across time without joining causal
claims. A complete chapter means its declared simulator evidence is connected;
it does not mean that the world model improved the score, that one intervention
is best, or that the result transfers to real football. No training, match,
formal experiment, human study or provider call is executed.

## 70. V3.52 Actionable and conflict-safe world chapter navigation

The season navigator initially described recent completed worlds, but the
decision ledger rendered the earliest six entries. A recent navigation chapter
could therefore have no visible detailed target, and index-based enhancement
layers could attach evidence to the wrong fixture after the display order
changed.

The browser now derives every ledger layer from one visibility function:
the current frozen decision, if present, followed by the six most recent
completed fixtures in reverse chronological order. Each rendered ledger card
receives the exact fixture identity as a data attribute and a programmatic
focus target. All later evidence layers consume the same ordered projection,
so their child-card alignment no longer depends on a separate slice.

Each visible history chapter now offers an explicit “open complete world
chapter” action. It resolves a card by exact fixture-identity equality without
constructing a CSS selector, expands its progressive evidence sections, scrolls
to it and transfers keyboard focus. If the authoritative card or the current
primary-action target is absent, disabled or has no actionable descendant, the
interface announces an error instead of silently doing nothing.

The Python projection also rejects empty or duplicate fixture identities before
constructing history. This prevents two canonical-looking chapters from
competing for one browser target even if an edited ledger entry was rehashed.
The navigator remains read-only and reuses existing mutation controls.

This is navigation integrity, accessibility and product-cohesion work. It does
not run training, generate futures, execute matches, estimate intervention
effects or strengthen any causal claim.

## 71. V3.53 Complete navigation contract and world-chapter permalinks

The navigator projection already exposed optional secondary actions and current
workspace continuity gaps, but the first browser integration consumed only the
primary action. That made the canonical state richer than the visible product:
users could not see the explicit choice between further bounded exploration and
advancing the official world, and current evidence discontinuities disappeared
at the navigation boundary.

The command center now renders every projected secondary action in a labelled
control group while keeping exactly one visually dominant primary action. Both
paths use one target resolver. It accepts only an existing, enabled control
outside a hidden subtree, resolves container targets to an enabled interactive
descendant, scrolls and focuses that control, and announces a failure when no
safe target exists. The current intervention chapter also displays either its
declared continuity gaps or an explicit no-known-gap state.

Completed world chapters now have reloadable browser locations. Opening a
chapter writes a URL fragment bound to both the canonical season identifier and
fixture identifier, expands all progressive evidence sections, marks the
ledger card as the current location and updates each history button's pressed
state. Page refresh and browser history traversal restore the same chapter.
History traversal rebuilds both the ledger and navigator from the current
season projection before resolving the fragment, so moving between two older
chapters outside the recent-six window remains reversible.
The resolver compares decoded identifiers as strings and never interpolates
them into a CSS selector.

Permalinks remain valid after a chapter falls outside the default recent-six
view: the selected completed ledger entry is added to the visible projection
without displacing the current frozen decision or recent history. A link from
another season, an unknown fixture, an overlong or malformed fragment, and a
pending decision all fail closed with visible feedback. Ledger cards therefore
carry separate fixture-identity and completed-chapter markers.

The navigator builder now also validates the current workspace continuity-gap
list instead of merely copying it. Tests cover preservation, invalid values,
secondary-action uniqueness and non-empty targets. Architecture audit tokens
require the server projection, safe target resolver, season-bound URL state and
cross-season rejection to remain connected.

This stage makes the complex-system navigation contract complete and
shareable. It does not create a second state authority, add a mutation route,
run training or matches, call a provider, estimate outcome effects, or claim
that a navigable chapter proves intervention quality.

## 72. V3.54 Whole-season validation and influence-path coverage

The season navigator retained at most 64 historical chapters. Its first
implementation projected and identity-checked only that retained window, while
the completed count included every ledger entry. It also labelled runtime-chain
and local-action totals without stating that those values came only from the
visible window. A corrupted older entry could therefore remain outside
navigator validation, and a long season could present window counts as if they
were season totals.

The navigator now projects and validates every completed ledger chapter before
applying the history limit. Entry, thread, six stage and action-evidence
identities are checked for old and recent fixtures alike. Only after this
validation does the product reverse and retain the most recent 64 chapters for
navigation. A test tampers with the oldest chapter beyond the retained window
and verifies that construction still fails closed.

Summary fields now have explicit scopes. Completed chapters, official runtime
chains, complete world-model runtime chains, local action changes and
continuity-gap chapters are whole-season totals. Parallel visible-prefixed
fields describe the retained 64-chapter navigation window. Truncation remains
explicit and the navigator identity seals both scopes.

The same validated whole-season chapters derive a six-part world-model
influence path:

1. a prematch review entered the final fixture selection;
2. the selected tactic was verified in the official runtime;
3. official world-model action evidence was retained;
4. at least one simulator-local action change was observed;
5. a persistent world-state transition was retained;
6. the complete world-model runtime chain was connected.

The Web command center presents every item against the same completed-fixture
denominator. These are independent evidence-coverage counts, not a monotonic
funnel: a persistent team-state transition is an official simulator fact and
is not attributed to the world model merely because both occur in one chapter.
The panel also surfaces the total number of chapters with declared continuity
gaps.

This correction improves evidence integrity and makes the world model's actual
product participation legible. It does not assert that action changes improved
decisions or scores, rank interventions, run training or matches, or authorize
real-football causal transfer.

## 73. V3.55 Season evidence-break diagnostics

The influence path made coverage visible but still required users to open
individual chapters to understand why a chain was incomplete. The navigator
summary reported only the number of affected chapters, which could not
distinguish a missing final review from absent runtime action evidence or a
missing persistent-state snapshot.

The whole-season navigator now aggregates every historical continuity-gap code
as a chapter count. Counts are ordered by descending affected chapters and then
by the gap code, giving canonical output independent of input dictionary order.
The summary also records completed chapters without declared gaps. A chapter
or current workspace with the same gap repeated twice is rejected, preventing
one fixture from inflating its diagnostic count.

The existing influence panel contains a progressive-disclosure diagnostic
rather than another top-level product surface. It localizes the five canonical
historical gaps:

- final review did not reach the frozen decision;
- official tactical runtime binding is unavailable;
- official world-model action evidence is unavailable;
- the descriptive match result is unavailable;
- the persistent postmatch or recovery state is unavailable.

Each row reports affected fixtures, preserves the technical gap code and gives
a bounded inspection suggestion. Unknown future codes remain visible with a
generic instruction to inspect the chapter evidence timeline; they are never
dropped or interpreted as a known failure. A completed season with no gaps
shows an explicit no-declared-gap state.

These diagnostics explain evidence availability, not system effectiveness.
They do not repair historical evidence automatically, identify a causal fault,
claim that a complete chain improved the result, execute a match, run training
or call an external provider.

## 74. V3.56 Evidence-break to chapter navigation

The season diagnostic identified break types and inspection guidance but left
users to search the historical ledger manually. This was especially poor for a
rare older break outside the recent-six cards: the system knew the affected
chapter identity but did not expose a direct evidence path.

Each aggregated gap now binds its most recent affected fixture, matchday and
canonical chapter identity. Selection is deterministic by matchday and fixture
identifier rather than relying on map order. Historical matchday values are
validated as positive integers before they may participate in this binding.
The full navigator identity seals the diagnostic count and its chapter target.

Every diagnostic row exposes an “open most recent affected chapter” button with
an explicit accessible name. Before navigation, the browser verifies both:

1. the fixture still exists as a completed decision-ledger entry;
2. the navigator summary still contains the exact fixture and chapter-identity
   binding supplied by the diagnostic.

If either check fails, navigation stops with visible stale-binding feedback.
When both pass, the existing season-and-fixture URL fragment is written, the
ledger and navigator are rebuilt from current state, and the selected chapter's
full progressive evidence is expanded and focused. This also works when the
target lies outside the default recent history window.

The action is read-only. It does not mutate a decision, repair a break, rerun a
match, infer a cause, train a model or call an external provider.

## 75. V3.57 Whole-season world-model action-adoption ledger

The influence path established whether world-model evidence reached each
completed fixture, but it still compressed the decisive mechanism to one
number: local action changes. That did not reveal the denominator of retained
manager-team decisions, how often the model changed action probabilities,
whether a changed sample was realized, or how much evidence reached an exact
ball-event identity. A complete chain could therefore remain legible while the
world model's practical action adoption was still opaque.

The official world-evolution thread now retains the complete bounded action
accounting needed by the product: retained and resolved decisions,
non-zero-probability influence, attribution-eligible decisions, locally
attributable changes, direct and changed ball-event links, decision-only
records, unresolved links, source opportunities and truncation state. These
facts are copied from validated official-match evidence and sealed into the
stage, thread, ledger entry, chapter and navigator identities. The navigator
rejects negative or boolean counts, impossible subsets, a broken event-link
partition, retained records beyond source opportunities, contradictory
evidence states, and an inconsistent coverage/truncation flag even when outer
identities have been recomputed.

Every completed chapter receives one descriptive action-adoption state:

- a shared-randomness local action change was realized;
- probabilities changed but the sampled action did not;
- no non-zero action influence was observed;
- stable mode made the action layer inapplicable; or
- official action evidence was unavailable.

The whole-season ledger reports the authoritative record totals, evidence and
coverage fixture counts, influence rate, realized-change rate among influenced
decisions, and resolvable ball-event link coverage. A zero denominator remains
`null` rather than being presented as zero performance. State rows are ordered
canonically and bind their most recent fixture, matchday and chapter identity.
The browser joins those bindings with the existing continuity-gap bindings and
uses the same completed-fixture, current-identity, permalink, rebuild, expand
and focus contract for drill-down. A stale state diagnostic therefore cannot
open a chapter merely because its fixture identifier still exists.

The account appears inside the existing season-world panel instead of creating
a detached analytics page. It distinguishes probability influence from a
realized action change and from event observability. These are simulator-local
mechanism and provenance facts only: the ledger does not compare scores,
estimate outcome effects, rank tactics, train a model, execute a match or
authorize real-football causality.

## 76. V3.58 Descriptive action-to-world propagation

The action-adoption ledger answered whether the world model influenced and
changed official simulator decisions, but a user still had to leave that
summary and inspect separate ledger paragraphs to see what the same completed
world recorded afterward. This preserved evidence integrity but weakened the
product's central complex-system story: intervention, action adoption, match
fact and persistent team state remained visually adjacent rather than one
navigable chapter.

Every historical chapter now projects one identity-sealed
`descriptive_world_after` record. It joins the already manager-oriented
descriptive outcome and points to the authoritative post-match team-state
transition: fatigue, morale, media pressure, availability-count deltas,
changed players, new and cleared injuries, new and cleared suspensions, and
whether matchday recovery is complete. The projection validates exact metric
and summary key sets, finite non-boolean numbers, non-negative integer event
counts, the outcome-to-points identity, the descriptive-only result boundary,
result and transition source identities, and transition-status/recovery
consistency. Recomputed outer hashes cannot legitimize a contradictory result
or malformed persistent-state projection.

The whole-season action ledger aggregates these facts once overall and once
inside each canonical action-adoption state. It reports raw chapter coverage,
win/draw/loss counts, points, summed match-state deltas and transition-event
counts. Zero evidence remains visible as zero coverage; no missing result or
state is imputed. The nested record explicitly fixes
`descriptive_cooccurrence_only`, rejects state-effect comparison, contains no
outcome estimate and authorizes no causal claim.

Studio extends the existing adoption-state cards and recent world chapters
rather than adding another dashboard. Each state shows its same-chapter later
facts and retains the V3.57 identity-bound jump to the most recent matching
chapter. The interface never ranks states or describes their raw differences
as effects. This makes the product chain legible from intervention through
action and into the persistent simulated world while preserving the boundary
between chronology, co-occurrence and causality.

This stage executes no match, training, formal experiment or provider call.
It does not repair missing evidence, infer why a result happened, compare
counterfactual outcomes or transfer simulator observations to real football.

## 77. V3.59 Identity-bound season world trajectory

The previous world navigator exposed recent chapters and whole-season totals,
but not the temporal path between them. A manager could inspect individual
matches or grouped action-adoption states without seeing how verified results,
local action changes and persistent team-state transitions accumulated across
the season. That kept the product legible as evidence infrastructure but made
the continuous football world feel unnecessarily fragmented.

The navigator now deterministically replays every completed decision-ledger
chapter into one chronological trajectory. Ordering uses matchday and fixture
identity, completed matchdays must be unique, and every visible point binds the
source fixture and canonical chapter identity. Each point preserves the action
adoption state, locally attributable simulator action changes, descriptive
result and points, recovery status, exact match-state deltas, transition-event
counts and explicit evidence coverage. It also carries season-to-date totals
for known results, points, persistent-state chapters, action changes, metrics
and transition events. Floating totals are rounded after every accumulation so
the replay identity remains deterministic.

The trajectory retains at most the same 64 recent points as chapter history,
but cumulative values are calculated from all completed chapters before that
display window is applied. The response declares the total and visible point
counts and whether truncation occurred. A missing result or persistent-state
record is never filled from a neighbour: the corresponding point is marked
unavailable, coverage does not advance, and `missing_evidence_imputed` remains
false.

Deterministic world-change markers make visible where the record contains a
local action change, new injury, new suspension, recovery still pending or an
evidence gap. These markers are observations, not an automatically inferred
turning point. Both the trajectory and every identity-sealed point explicitly
disable turning-point inference and causal authorization, and the trajectory
contains no outcome-effect estimate.

Studio renders the chronology inside the existing season-world navigator as a
progressive list. Each row shows same-match facts beside season-to-date totals
and uses the existing completed-fixture permalink flow. Before opening a row,
the browser verifies the exact fixture-and-chapter identity against the current
trajectory, so a stale or substituted binding fails closed. Selection state is
synchronized between the full ledger, recent chapter cards and trajectory.

This stage is code-only. It executes no match, model training, formal study or
external provider call, and it does not claim that a temporal association is a
world-model effect or a real-football causal relation.

## 78. V3.60 Reviewed futures carried into the official world

The counterfactual workbench could generate and review fixed-budget,
multi-timepoint simulator futures, and the official world thread could prove
that a reviewed selection reached runtime. However, the hand-off collapsed the
reviewed evidence to a boolean-like state and a mechanism-example count. Once
the official match became a historical chapter, users could no longer see how
many branch times were eligible, whether actions diverged, whether local
attribution existed, whether descriptive downstream windows differed, whether
timing sensitivity was observed, or whether the manager kept or revised the
selection. The product therefore still separated "possible futures" from the
world that actually continued.

The terminal future-review execution trace now carries the validated review
intent, evidence level, fixed scenario budget, eligible and verified anchors,
action-divergence scenarios, locally attributable scenarios, descriptive
future-difference scenarios, timing-sensitivity observation and the explicit
no-ranking result. These fields are copied only from a validated review receipt
and are sealed by the trace identity. The prematch stage of the canonical world
thread copies the same facts, and its stage and thread identities seal the
review-to-world hand-off.

The season navigator independently checks the trace identity, exact trace-to-
stage equality, non-negative integer counts, fixed-budget bounds, eligible
anchor coverage and the local-attribution subset of action divergence.
Descriptive future difference remains an independent observation, matching the
source scenario contract instead of being forced into a false funnel. A
rehashed trace and rehashed thread cannot authorize an impossible partition.
Each official chapter and trajectory point then retains one
`reviewed_future_context`. Whole-season trajectory totals distinguish
reviewed chapters, reviews selected for the fixture, retained per-scenario
evidence, reviewed action divergence and reviewed timing sensitivity.

This audit also corrected a temporal-window defect. The authoritative decision
ledger is latest-first, while early navigator code implicitly treated its input
as chronological before taking a bounded slice. All completed chapters are now
sorted by matchday and fixture identity before recent-history truncation,
aggregation and trajectory construction. Tests reverse the source ledger and
prove that the same latest 64 chapters and chronological trajectory survive.

Studio displays reviewed branch facts inside the existing prematch world-thread
stage and beside the corresponding official trajectory point. The influence
path reports their fixture-level coverage as independent evidence counts. A
dedicated boundary states that this continuity records what the manager saw,
whether the decision changed and whether it reached runtime; it does not match
simulated paths to the observed score as forecast accuracy or treatment effect.

This stage executes no future generation, match, training, formal study or
provider call. It introduces no best-time recommendation, outcome comparison,
score prediction, intervention ranking or real-football causal authority.

## 79. V3.61 Identity-sealed historical fork dossiers

V3.60 retained the aggregate context of a reviewed future set after the
official world advanced, but still discarded the concrete roads not taken. A
historical chapter could say that three times were reviewed and two contained
action divergence without preserving which times, which branch identities or
which local/descriptive differences produced those totals. This left the
season trajectory continuous at fixture level but opaque at fork level.

The terminal review trace now derives a bounded `reviewed_scenarios` archive
from the already validated scenario evidence. It retains exactly the frozen
2--4 branch times and only compact facts: source scenario identity, time,
semantic status, eligibility and anchor state, branch-state identity, changed
and locally attributable action counts, descriptive future-difference count,
and both causal-denial flags. Mechanism examples remain in their existing
detailed evidence surface and are not duplicated. Every compact record gets a
new deterministic `archive_identity`; the enclosing review trace identity
seals the ordered list.

The canonical prematch world-thread stage copies the archive exactly. The
navigator then independently validates field closure, finite ordered times,
minute conversion, source/archive identities, anchor and eligibility
semantics, count types and bounds, local-attribution semantics, status
derivation and both causal boundaries. It recomputes all five aggregate counts
and timing sensitivity from the concrete archive. Consequently, rehashing an
outer trace or thread cannot make altered aggregate facts consistent with
unchanged fork records. Aggregate-only legacy reviews must carry an empty
archive and are never upgraded to scenario evidence.

Official world chapters and identity-bound trajectory points carry the exact
validated archive. Whole-season counts distinguish retained dossiers from
identity-verified dossiers. Studio renders the records as progressive details
inside the existing prematch thread and the matching trajectory row, showing
the branch minute, semantic status, action/local/descriptive counts, anchor
verification and shortened source/archive identities. No separate dashboard
or competing navigation model is introduced.

These dossiers describe simulated roads the official world did not
simultaneously experience. They are not ranked, matched to the observed score,
or treated as prediction accuracy, intervention effects, counterfactual truth
or real-football causality. This stage is code-only and executes no future
generation, match, training, formal study or external provider call.

## 80. V3.62 Review-to-official-world continuity certificate

The prior world thread used `world_model_runtime_chain_complete` to mean that
an official match retained tactical runtime, world-model action, result and
persistent-state evidence. That was useful but insufficiently specific: the
same flag could be true when no prematch future was reviewed, so it could not
prove that the manager's reviewed intervention was the decision that entered
the official simulator world.

Every world thread now derives one identity-sealed
`review_to_official_world` certificate. It binds the terminal review and trace
identities, final decision identity, tactical binding identity, official
action-evidence identity, non-empty same-match identity and retained scenario
count. Its state is exactly one of `complete`, `awaiting_official_match`,
`review_superseded`, `tactical_runtime_binding_unavailable`,
`official_action_evidence_unavailable` or `not_reviewed`. The separate
`reviewed_world_model_chain_complete` flag is true only for `complete`; the
older runtime flag keeps its narrower official-execution meaning.

The navigator independently validates the certificate's closed field set,
hash, every source identity, scenario count, boolean semantics and state
derivation. Two missing match IDs cannot satisfy same-match continuity. The
certificate is copied into the canonical chapter and trajectory point, while
season summaries count reviewed-world completion separately from ordinary
world-model runtime coverage. Rehashed attempts to authorize scenario/runtime
matching or contradict a source stage fail closed.

Studio shows the certificate inside the existing six-stage world thread,
beside its trajectory point and as the final evidence-coverage item. This
closes the product question “did the reviewed decision reach the official
world?” without inventing a second workflow.

The certificate explicitly fixes
`scenario_to_runtime_opportunity_matching_performed` to false. Prematch forks
and official match actions have different trajectory contexts and are not
declared to be identical opportunities. Complete continuity therefore proves
identity and runtime adoption only; it is not scenario accuracy, action-level
forecast validation, score improvement, treatment effect or real-football
causality. This stage executes no future generation, match, training, formal
study or provider call.

## 81. V3.63 Validated multi-action probability authority

The unified match engine has always sampled pass, shot, cross and hold, but the
direct world-model probability controller previously read only the pass gate.
Shot imagination could change an auditable planner utility while remaining
unable to change the actual categorical sampling distribution. This was a
real authority bottleneck: a four-action engine had a pass-only learned policy
interface.

The direct controller now operates on the complete feasible-action simplex.
Every open action-specific gate contributes a finite model advantage and an
independently bounded authority derived from its policy budget, decision
confidence and decision certainty. The strongest authorized signal controls
the overall interpolation budget (still capped at 0.35); relative authorities
shape a normalized target distribution. A single shared uniform draw continues
to sample the baseline and treatment distributions, preserving local
counterfactual attribution without applying planner utility deltas twice.

Shot planning now exposes its validated shot-versus-continuation advantage,
context attenuation, certainty and shot-specific blend to that controller.
An infeasible shot is explicitly closed before scoring. At this stage cross
remained closed with `no_action_specific_validation` because no suitable
validation artifact existed; the implementation did not borrow pass evidence
to manufacture cross authority. Hold remains the reference continuation action unless a future
action-specific contract authorizes direct promotion.

Runtime evidence originally declared `validated_action_simplex_v1`, lists the exact
policy actions admitted for each opportunity and aggregates action-level
positive/negative guidance, probability movement, applied authority and local
changes. The compact Studio status preserves this bounded breakdown and shows
per-action signal cards. These are runtime observations, not claims of match-
outcome improvement.

Changing either the controller or planner invalidates the execution identity
of the previously sealed mechanism and full-match studies. Studio now replays
the protocol, checkpoint and every declared code hash before publishing a
formal result. The old result remains identifiable as historical evidence but
its statistics, mechanism payload and promotion fields are suppressed for the
current controller, with `stale_current_code_identity` shown instead. This
prevents evidence collected for the pass-only controller from being silently
transferred to the new pass-and-shot authority surface.

This stage is code-only. It runs no training, match, fixed-budget experiment or
provider call. Existing formal artifacts remain immutable historical records,
but are not current-code evidence. Broader validated action authority makes
the new mechanism testable, but does not itself establish effectiveness or
real-football causality.

## 82. V3.64 Cross-specific evidence and planning authority

V3.63 made the action controller multi-action, but cross still had no learned
evidence path. Cross execution was absent from ball trajectories, training
backfill accepted only pass/shot/intercept, online calibration grouped every
non-shot transition with pass, and the runtime had no cross-specific quality.
Opening the cross gate in that state would have reused unrelated pass evidence.

Cross now has a closed evidence chain. Executed crosses are logged before the
pending policy identity is cleared, with crosser, normalized origin and landing
point, aerial contact, winner, added simulator xG, goal state, trajectory peak
and flight time. Optional pre/post observations use the same snapshot contract
as pass and shot. The action codec and ball-log dataset preserve cross as action
index three, while Studio replay renders it as a distinct dashed trajectory and
links it to a policy record only by exact opportunity identity, action, team and
timestamp. Only hold remains trajectory-free by design.

Future training runs compute a grouped-heldout cross validation artifact from
cross rows in the existing validation split. The registered metric is weighted
next-observation MSE against same-state persistence. Eligibility requires at
least 96 cross transitions across six match groups and at least 0.02 skill over
persistence. Quality is support-adjusted and recomputed from primitive counts
and losses whenever a checkpoint loads; incomplete, relabelled or tampered
evidence yields exactly zero authority. The scope is simulator transitions
only and explicitly carries no external-football validity.

The runtime separates pass, cross and shot online-calibration branches. A valid
cross checkpoint may compare a learned cross future with the learned hold
continuation, apply the cross-specific confidence and certainty budget, and
send the bounded advantage to the shared feasible-action probability simplex.
Cross does not consume the pass-success head. The blend is independently capped
by `MATCH_WM_CROSS_BLEND` and remains subordinate to the controller's global
0.35 ceiling.

`cross_action_validation_protocol_v1.json` and its read-only verifier make the
current state explicit. The existing v9 candidate hash is valid but its metadata
predates the cross artifact, so verification reports
`blocked_checkpoint_missing_cross_validation`, quality 0.0 and no cross
planning authority. Studio surfaces "unvalidated, zero authority" instead of
confusing code readiness with evidence readiness. No old checkpoint is upgraded
and no historical match result is reinterpreted.
The verification snapshot seals the protocol, checkpoint and every required
code file by SHA-256. Studio recomputes that identity before displaying the
snapshot; any later code drift changes the state to
`stale_current_code_identity` and forces code readiness, authority and quality
to fail closed.

This stage executes no training, match, fixed-budget experiment or provider
call. It completes the code and evidence contract needed for a later checkpoint;
it does not demonstrate cross prediction skill, behavior change, match-outcome
improvement or real-football causality.

## 83. V3.65 Sealed-only shot action authority

The checkpoint carried a development-split `shot_planner_quality`, while the
formal sealed evaluation separately showed that its joint shot head was worse
than the physics xG prior. Runtime happened to keep the current head closed
because its quality was low, but the implementation still allowed a future
high development score to authorize the joint head without sealed evidence.
Its reported probability source also said `joint_world_model_head` even when
the product decision retained physics xG. This was a latent authority and
truthfulness defect.

Runtime now preserves `joint_shot_quality` as a diagnostic only and initializes
authorized `shot_quality` to exactly zero. Without an independently promoted
`FrozenShotHead`, shot probability is the leakage-stripped action's physics xG
prior and the source is `physics_xg_prior`. The joint outcome head cannot open
the shot action gate regardless of its development score. A frozen head can
replace that fallback only after its existing contract verifies the exact base
checkpoint, sealed-only samples and goals, proper-score gain over physics xG,
and an accepted promotion artifact.

The shot scorer now returns an actual goal probability instead of a weighted
mixture of progress, xG and head probability that downstream code mistakenly
treated as a probability. Its decision uncertainty comes from the same
action-specific authority and online-trust path used by the planner. Studio
projects the sealed sample count, goal count, skill versus physics xG, current
probability source and frozen-head authority before any match runs. For the
current candidate this reads 49 sealed shots, eight goals, skill -0.00496,
physics xG fallback and zero learned shot authority.

This is a code and evidence-semantics correction only. It performs no training,
sealed evaluation, match, provider call or promotion, and it makes no claim that
physics xG is optimal or that a future frozen head will improve match outcomes.

## 84. V3.66 Reference-only hold attribution

Hold is the continuation reference used to ask whether a validated pass, shot
or cross is better than retaining the ball. It does not currently have an
independent grouped-heldout validation contract. The V1 simplex correctly left
its quality gate closed, but a suppression-only signal could still move
probability mass from pass to hold, label hold as `recommended_action`, and
count an observed hold as direct adoption. That confused a normalization effect
with learned hold authority.

`validated_action_simplex_v2` makes the distinction executable. The controller
admits direct signals only for pass, shot and cross, even if stale or forged
runtime evidence attempts to open a hold gate. The planner marks hold as
`counterfactual_reference_only`, with zero direct authorization. A positive
validated action can still be a direct recommendation. When all model evidence
is suppressive, `recommended_action` is `none`; the suppressed action remains
the primary signed signal, while any increased hold probability is recorded as
an indirect redistribution recipient.

The resolved record and aggregate diagnostics now separate direct adoption from
the hold reference effect. They retain the hold probability gain, realized hold
actions and locally changed-to-hold samples without adding any hold entry to the
action-specific signal breakdown or direct adoption count. Studio preserves and
shows this compact reference block with an explicit warning that it is not an
independent learned hold recommendation.
Internal utility movement behind a closed gate is diagnostic only and no longer
increments `influenced_opportunities`; influence begins only when the shared
probability controller accepts a finite, authorized direct signal.
The legacy mean-recommended-probability metric now accumulates only direct
positive recommendations. A separate mean-primary-signal metric retains the
magnitude of both promotion and suppression without relabelling suppression as
recommendation.

This corrects attribution semantics; it does not add a hold model, establish
that holding is optimal, or prove downstream outcome improvement. No training,
match, formal experiment or provider call is executed in this stage.

## 85. V3.67 Unified official action explanation chain

Cross trajectories became replayable in V3.64, but the official manager action
projection still accepted direct ball-event identity only for pass and shot.
It simultaneously allowed cross to use the hold-only
`no_ball_trajectory_by_design` status. A correctly linked official cross could
therefore be rejected at the product boundary even though the replay layer had
already verified its identity, action, team and timestamp.

The official projection now accepts pass, shot and cross only through exact
direct runtime identity links. Hold is the sole trajectory-free action. Tests
cover both a valid identity-bound cross and a forged trajectory-free cross, so
the product cannot silently regress to the pre-V3.64 contract.

The projection also carries the V2 policy version, signal mode, primary action,
primary probability delta and bounded hold-reference effect into each retained
manager-team example. Current records distinguish direct preference,
suppression-only and no-signal states. Legacy unversioned reports remain
readable as explicitly unclassified evidence rather than being reinterpreted.
The standalone match report and manager timeline render the same distinction:
probability redistributed to hold is shown as a reference effect, never as a
learned hold recommendation.

Every projected field remains inside the existing evidence identity and replay
validator. It proves only official simulator action selection and exact local
ball-event linkage. It does not convert downstream trajectories, tactics,
scores or persistent world state into causal outcome claims. This stage runs no
training, match, formal experiment or provider call.

## 86. V3.68 Cross-aware counterfactual propagation

The replay and official execution paths recognized cross after V3.67, but the
paired-world comparison still filtered replay events to pass and shot. Its
runtime matcher also lacked cross, and descriptive future windows folded no
cross count into their action summaries. A cross could therefore be directly
observed in an official report and disappear when the same evidence entered a
multi-time future set.

Paired comparison now retains pass, shot and cross events, matches cross by the
same opportunity identity, team, timestamp and action contract, and counts
`crosses` separately from `passes` in every 30- and 120-second descriptive
window. Tests establish that an identity-bound cross receives simulator-local
attribution and contributes one cross to the downstream window without
granting that window causal status.

Mechanism examples and their windows now use nested schema version 2. V2 adds
the cross metric plus probability-policy version, signal mode, primary signal,
and bounded hold-reference redistribution fields to the hashed example.
Validation still accepts version-1 examples with the original five window
metrics, so persisted historical future reviews remain replayable. New Studio
views surface non-zero cross windows and use the same direct-preference,
suppression-only and reference-only vocabulary as official match review.

This closes a product continuity gap; it does not claim that a cross caused a
shot, goal or result. Downstream windows remain shared-clock descriptions only.
No training, match, future generation, formal experiment or provider call is
executed in this stage.

## 87. V3.69 Aggregate action-semantic continuity

V3.68 preserved cross and policy-signal semantics inside individual paired
decisions and future mechanism examples, but two aggregate boundaries still
discarded them. The synchronized paired-world overview omitted cross from its
event totals, and the season decision ledger reduced retained mechanisms to a
generic example count. A product user could therefore inspect one detailed
record and see a cross or suppressive signal, then lose that distinction in
the comparison overview, season summary and world-evolution thread.

The paired replay now counts cross independently and its descriptive-window
table renders the same five metrics carried by the underlying schema. The
future-review execution trace and season ledger aggregate V2 semantics into
explicit counts for cross actions, direct-preference signals,
suppression-only signals, hold-reference redistribution and non-zero cross
windows. The manager world thread carries those counts forward, and Studio
shows them in both the season ledger and the prematch review stage. Version-1
mechanism examples remain valid and contribute zero to fields they never
recorded; they are not retroactively reclassified.

These are continuity and observability guarantees, not new model authority.
Cross windows remain descriptive, hold remains reference-only, and no
downstream or outcome causal claim is added. No training, match, future
generation, formal experiment or provider call is executed in this stage.

## 88. V3.70 Historical action-semantic continuity

V3.69 carried cross and policy-signal semantics into each official manager
world thread, but the season navigator still archived only the generic count
of retained mechanism examples. Once a completed fixture became a historical
chapter, cross actions, direct preferences, suppression-only signals,
hold-reference redistribution and non-zero cross windows disappeared from the
season trajectory. The history view therefore exposed less evidence than the
same fixture's live thread.

The navigator now normalizes, validates and identity-binds all five semantic
counts at the execution-trace-to-thread boundary. Counts must be non-negative,
must agree across both sources, cannot exceed the retained-example and bounded
window budgets, and survive into each chapter, chronological trajectory point,
whole-season trajectory totals and the world-model influence path. A forged
count still fails after outer identities are recomputed. Historical records
that genuinely predate the V2 fields normalize missing values to zero; explicit
invalid values or disagreements do not receive that compatibility treatment.

Studio renders the whole-season totals and the per-match semantic record next
to the existing reviewed-future chronology. This preserves product continuity
without ranking interventions or relating the counts to scores. No new action
authority, downstream causality or real-football claim is introduced. No
training, match, future generation, formal experiment or provider call is
executed in this stage.

## 89. V3.71 Bounded official-action semantic ledger

The official action projection already retained policy-signal, cross and hold
reference semantics on each displayed example. Its world thread and season
action-adoption ledger nevertheless reduced those examples to generic record,
influence and identity-link totals. This was especially risky because official
action evidence retains at most five explanatory examples: treating their
composition as the distribution of every retained decision would overstate the
evidence whenever the example list was truncated.

The official world thread now derives a separate bounded-example contract. It
records the retained semantic-example count, complete-coverage and truncation
flags, actual cross examples, direct-preference, suppression-only, no-signal
and legacy-unclassified partitions, hold-reference redistribution, and exact
direct cross ball-event links. The four signal-mode counts must partition the
retained semantic examples exactly. Cross and reference counts cannot exceed
the sample, direct cross links cannot exceed either sampled crosses or all
direct links, and complete coverage is true only when the sample equals the
official retained-record count without truncation.

Historical chapters preserve this bounded contract. The season adoption ledger
aggregates it under `bounded_semantic_examples`, and each chronological point
retains its own sample and coverage state. Both data and Studio explicitly set
`full_record_distribution_authorized` to false and label the values as bounded
samples. The live world thread, season summary and per-match trajectory expose
the same boundary.

This improves explanation coverage, not action authority or outcome evidence.
The sample cannot establish the full action distribution when any fixture is
truncated or lacks semantics, and no signal count is compared with scores. No
training, match, future generation, formal experiment or provider call is
executed in this stage.

## 90. V3.72 Full retained-record action semantics

V3.71 deliberately treated the at-most-five displayed examples as a bounded
explanation sample. That boundary was correct, but it also meant the product
could not answer a separate factual question: across every retained official
manager action record, which actions occurred and which world-model signal
mode was present? Increasing the example cap would have mixed explanation
sampling with statistical accounting and still failed whenever source manager
records were truncated.

The official action projection is therefore versioned to V2 while retaining
strict V1 read compatibility. V2 derives a compact aggregate directly from all
records retained by the official manager-action report, independently of the
bounded example list. It partitions actual actions and primary signal actions
across `hold`, `pass`, `cross`, `shot` and `none`; partitions signal modes
across direct preference, suppression-only, no signal and legacy-unclassified;
and separately counts hold-reference redistribution, direct cross ball-event
links and locally attributable cross changes. Every partition must equal the
retained-record total. Cross-specific counts remain bounded by the corresponding
official action and identity-link totals.

Coverage semantics remain explicit at both boundaries. The aggregate always
covers every retained record, but it authorizes a full *source* distribution
only when the upstream manager record set was not truncated. A V1 payload may
omit the aggregate; it may not smuggle V2 fields into the legacy schema. When
the bounded examples happen to cover every retained record, validation
independently recomputes the aggregate from those examples and requires exact
agreement. Rehashing an internally inconsistent aggregate therefore still
fails closed.

The identity-bound aggregate now survives the decision ledger, live world
thread, historical chapter, chronological trajectory and season action-adoption
ledger. Studio renders the same distinction in official action details, the
decision summary, world thread, history cards, trajectory points and season
totals. Bounded examples remain available for explanation; the V2 aggregate is
the authoritative retained-record composition. Neither representation grants
outcome attribution, ranks interventions or establishes real-football effects.
No training, match, future generation, formal experiment or provider call is
executed in this stage.

## 91. V3.73 Counterfactual action-transition ledger

V3.72 established the composition of every retained official action record,
but composition alone could still hide the mechanism that matters to a
manager. Knowing that one pass and one hold occurred does not reveal whether
the world model changed hold into pass, pass into hold, or changed nothing.
The decision-ledger UI also read the season summary's retained count from a
non-existent `records` field instead of
`retained_records_with_v2_semantics`, causing a valid aggregate to display as
zero.

Official action evidence is now V3. Its nested retained semantic contract is
version 2 and adds two complete 5-by-5 matrices over `hold`, `pass`, `cross`,
`shot` and `none`: all counterfactual-baseline-to-actual transitions, and the
subset satisfying shared-uniform simulator-local attribution. Matrix totals
must equal the retained-record and locally attributable change totals,
respectively. All-transition column marginals must exactly reproduce the
actual-action partition. Local transitions cannot exceed their corresponding
all-transition cells, occupy a diagonal cell, or involve the unresolved
`none` state. These constraints are checked before outer identity verification,
so recomputing only the envelope hash cannot legitimize a forged matrix.

V1 official evidence remains readable without semantic aggregates. V2 remains
readable with the V3.72 nested semantic contract and is never reclassified as
having transition evidence. V3 requires the new matrix contract. Decision
ledgers and historical navigators aggregate only genuine V3 transition rows,
publish explicit missing-version counts, and authorize a full source transition
distribution only when every contributing fixture is V3 and no upstream source
record was truncated.

Studio now renders non-zero locally attributable transitions in manager terms,
such as `持球→传球`, in official action details, live threads, history cards and
trajectory points. The decision summary uses the correct retained-record field
and separately reports transition-matrix coverage. This identifies how the
world model changed simulator actions; it does not establish that the change
improved tactics, scores or real football. No training, match, future
generation, formal experiment or provider call is executed in this stage.

## 92. V3.74 Transition-stratified descriptive world propagation

V3.73 made local baseline-to-actual action changes explicit, while the season
navigator still exposed later match and persistent-state facts only by generic
adoption state. A manager could see `持球→传球` and separately see fatigue,
morale, injuries and the score, but could not navigate the seasons in which
that specific local transition occurred.

The navigator now derives a transition-stratified propagation ledger from
identity-validated V3 chapters. Each non-zero locally attributable transition
records its occurrence count, distinct source chapters, latest chapter
identity, source-transition coverage and the same-chapter descriptive facts:
available outcomes and points, persistent-state coverage, recovery completion,
metric deltas and injury/suspension summaries. V1 and V2 chapters contribute
to explicit missing-V3 coverage, never to invented transition strata.

The strata are intentionally not cohorts or experimental arms. One chapter may
contain several distinct local transitions and therefore enter several strata;
the ledger counts and exposes that overlap. It forbids cross-stratum effect
comparison, transition ranking and outcome attribution. Studio repeats these
boundaries, renders the descriptive facts for each transition and links to the
latest identity-bound world chapter. The feature answers “what was observed in
the same chapter after this simulator-local transition?”, not “what effect did
this transition cause?”. No training, match, future generation, formal
experiment or provider call is executed in this stage.

## 93. V3.75 Accessible manager action-transition map

V3.73 and V3.74 made action transitions queryable and connected them to later
world facts, but Studio still presented the mechanism primarily as prose and a
sequence of cards. That made a complete action policy difficult to scan and
made legacy absence easy to confuse with an observed zero.

Studio now renders the locally attributable baseline-to-actual matrix as an
accessible 5-by-5 table. Row headers are counterfactual baseline actions,
column headers are officially adopted actions, every cell has a spoken label,
and non-zero cells receive a redundant visual emphasis without relying on
color alone for their value. The table sits in a keyboard-focusable horizontal
scroll region so its full structure remains usable on narrow screens and under
zoom. Rendering uses only `createElement`, `textContent` and attributes; no
evidence value enters HTML parsing.

The map is shown only for completed manager-world chapters. If completed
chapters lack V3 matrices, Studio hides the empty table and says that evidence
is unavailable rather than claiming zero action influence. With V3 evidence it
reports covered fixtures, legacy or missing fixtures, local-transition count
and whether the source distribution is complete. A displayed zero therefore
means “not observed inside available V3 evidence”; it never means impossible,
ineffective or absent in real football. The matrix remains descriptive and
does not authorize action ranking, cross-cell effect comparison, score
attribution or real-football causality. No training, match, future generation,
formal experiment or provider call is executed in this stage.

## 94. V3.76 Exact shared-uniform action-change expectation

The action controller samples the baseline and world-model policies by feeding
the same uniform draw through two categorical inverse CDFs. Earlier diagnostics
summed total-variation distance and named that value “expected counterfactual
action changes.” TV is the minimum mismatch probability over all possible
couplings; it equals the shared-draw mismatch for two actions, but not generally
for the simulator's three- or four-action distributions. For example, ordered
policies `(0.5, 0.5, 0.0)` and `(0.0, 0.5, 0.5)` have TV 0.5 while their
same-label inverse-CDF intervals do not overlap, so the actual shared-draw
mismatch probability is 1.0.

The controller now computes that probability exactly by summing overlap between
same-action CDF intervals and subtracting it from one. Both high-level action
and executable pass-target expectations use the same estimator. Each retained
runtime record keeps total variation as a separate distribution-distance
diagnostic and adds its exact shared-uniform change probability. Diagnostics
publish estimator identity `shared_uniform_inverse_cdf_overlap_v1`, and the
probability-policy contract advances to `validated_action_simplex_v3`.

Workspace projection preserves the estimator identity. Studio and HTML reports
label current values as shared-draw expectations, while old records without the
identity are explicitly shown as legacy proxies. This corrects measurement; it
does not increase model authority, alter an action distribution, fabricate a
realized change or authorize outcome claims. No training, match, future
generation, formal experiment or provider call is executed in this stage.

## 95. V3.77 Official V4 shared-draw expectation continuity

V3.76 corrected runtime measurement, but the official-match projector retained
only total variation. The exact shared-draw probability therefore disappeared
before the decision ledger, world thread and season navigator. A current match
could display a correct expectation in the latest-match panel while the same
match's authoritative manager history could not reproduce that value.

Official action evidence now supports V4. A runtime declaring
`validated_action_simplex_v3` must also declare the exact estimator and provide
a finite `[0, 1]` shared-uniform change probability for every retained record;
missing or malformed values fail closed. V4 retained semantics version 3 sums
only attribution-eligible exact probabilities, counts exact records and exact
eligible records, and distinguishes complete retained coverage from complete
source coverage. Its identity includes all these fields. When bounded examples
cover the retained set, validation recomputes the full semantic payload, so an
attacker cannot legitimize a modified expectation by merely rehashing the outer
envelope.

Historical behavior stays explicit. Sources without the exact estimator still
project as V3; stored V1, V2 and V3 evidence remains readable and is never
upgraded to V4. Decision and season aggregates publish fixtures with and without
V4 semantics, exact-record coverage, summed expected changes and a full-source
authorization flag. World threads, chapters and trajectories preserve the same
nested evidence. Studio shows exact expectation next to realized changes and
labels old chapters as unavailable rather than zero.

This closes a measurement and provenance chain only. It does not raise action
authority, change the sampled policy, convert probability mass into observed
events or claim tactical/outcome causality. No training, match, future
generation, formal experiment or provider call is executed in this stage.

## 96. V3.78 Action-transition to world-propagation drill-down

The season navigator already contained two correct but disconnected views: a
5-by-5 local action-transition matrix and descriptive same-chapter propagation
strata. It also filtered strata to retained semantic schema 2 only. Because V4
official evidence uses retained semantic schema 3 while preserving the same
transition matrices, a current exact-evidence chapter could disappear from the
downstream propagation view.

The propagation aggregator now treats retained semantic schemas 2 and 3 as
V3-compatible transition evidence. Mixed historical seasons keep their original
versions while contributing all valid transition matrices; missing old matrices
remain explicit coverage gaps. A regression fixture combines V3 and V4
chapters and verifies both are counted without upgrading either record.

Studio now turns each evidenced non-zero matrix cell into a keyboard-accessible
button. Selecting baseline-to-adopted joins by the stable transition id and
reveals its occurrence and chapter counts, same-chapter downstream world facts,
source-coverage boundary and a verified link to the latest relevant world
chapter. Zero cells and cells without matching propagation evidence remain
disabled. Selection uses aria-controls, aria-pressed, a labelled live region
and explicit focus movement, without HTML injection.

The detail is deliberately descriptive. A chapter may contain several local
transitions and therefore enter several strata; the UI prohibits ranking cells,
cross-cell effect comparison, score attribution and real-football causal claims.
No training, match, future generation, formal experiment or provider call is
executed in this stage.

## 97. V3.79 All-known credential closure and evidence invalidation

The prior security closure protocol registered only the exposed DeepSeek key.
It could therefore accept a valid DeepSeek revocation while remaining unaware
of the separately exposed GitHub Classic personal access token. A clean
repository scan was also limited to one OpenAI-compatible key pattern. That was
not sufficient evidence for the generic credential-security completion gate.

Security closure protocol V2 freezes two incident identities: the DeepSeek API
credential and the GitHub Classic PAT. The outer attestation must contain the
exact incident set. Every incident must be explicitly revoked, use a distinct
confined content-addressed redacted receipt, and record either no replacement
or local-environment-only replacement storage. One signed attestation binds the
complete set to the current commit. Partial closure, duplicate incident ids,
unknown or missing incidents, reused receipts, escaping paths, secret-bearing
evidence, stale scans and credentials embedded in the HTTPS origin URL all fail
closed. The origin check returns only a boolean and never prints the URL.

Repository scanning now reports only counts for OpenAI-compatible API keys,
GitHub Classic PATs and GitHub fine-grained PATs. Neither the verifier nor its
reports expose matched values. The evidence kit contains separate deliberately
invalid receipt templates and one deliberately invalid multi-incident
attestation template. Control-plane, completion-plan, reproduction-manifest and
excellence-roadmap paths all point to V2. The historical V1 protocol remains in
the repository as superseded evidence but cannot satisfy a current gate.

Refreshing this chain revealed an independent stale-evidence condition:
post-experiment action-adoption code had changed, while the old paper-package
snapshot still carried a passing status. The current paper-package verifier
correctly rejects its frozen execution identity. The control plane now reports
9 of 21 gates passed and 12 open while retaining the evidence-derived product
83 and academic 70 base maturity scores. Restoring the paper gate requires a
new explicitly authorized identity-matched experiment; this stage does not run
or rewrite that experiment.

No credential is used or stored. No provider call, network revocation, training,
match, future generation or formal experiment is executed in this stage.

## 98. V3.80 Identity-scoped stochastic runtime

Reproducibility previously depended on a mixture of named generators and the
process-global Python or NumPy random state. The principal scoring path already
had stable seeds, but referee sampling, social dialogue, agent latent updates,
physical wear, carryover injuries, world-day events and the legacy journey path
could still consume shared state. Adding an unrelated draw in one subsystem
could therefore change a later outcome in another subsystem while retaining the
same top-level seed.

The runtime now derives a stable match identity from the root seed, fixture
seed, stage and ordered teams. That identity is forwarded through preparation,
simulation, narrative and finalization. Named child streams isolate referee,
score, fusion volatility, dialogue, post-match latent state, physical wear,
medical reporting and carryover settlement. World-day activity selection,
per-agent behavior and headline matches are independent child streams, so a
social draw cannot perturb the day's match event. Bracket, black-swan, agent
initialization and legacy journey paths use identity-derived standard-library
streams as well. Public journey callers may provide a root seed; the default
remains deterministic for compatibility.

This is an isolation contract, not evidence of model quality. A static AST
audit now rejects direct process-global Python or NumPy random draws throughout
simulation and memory runtime modules. Tests verify replay under the same
identity, non-interference with global RNG state, environment-seed independence
once a match identity exists, and exact seed propagation across the four match
lifecycle phases. Local generator fallbacks remain available for isolated API
use, while the authoritative tournament and manager paths always inject named
streams.

No training, match, future generation, formal experiment or provider call is
executed in this stage. Existing frozen research results remain stale for the
current code identity and are not upgraded by this correctness change.

## 99. V3.81 Random-world identity across tournament resume

V3.80 isolated live stochastic subsystems, but the tournament checkpoint still
used schema V1. It stored match progress without the root seed or random-stream
contract. In addition, the public tournament and micro APIs seeded legacy
process generators without forwarding the requested seed into world
construction. A resumed process could therefore rebuild agents under a new
root seed and continue a valid-looking checkpoint in a different random world.

Checkpoint V2 stores `identity_scoped_rng_v1` and its integer root seed. A
canonical, finite-JSON SHA-256 checksum covers the complete payload except the
checksum field itself. Loading validates the version, checksum, random-world
identity and bounded structural types before exposing state. This checksum is
corruption/tamper detection, not a signature or proof against an actor able to
rewrite both content and digest.

The public resume path reads and verifies the checkpoint before constructing
the world. With no explicit seed it reuses the stored seed regardless of the
current environment; an explicitly different seed is rejected. The manager
also compares the loaded seed with its already constructed world, protecting
direct integration callers. Non-resume runs still use explicit seed, then
`GFS_SEED`, then 42. Public micro matches now pass their seed into agent/world
initialization as well as their match streams.

The run manifest is built from a captured runtime environment with the resolved
root seed overlaid, rather than from dataclass defaults. Score-path, cognition,
world-model and carryover flags in that manifest now describe the actual run
configuration at the construction boundary.

Tournament construction also forwards the caller's resolved project root into
the manager. Checkpoints, carryover state and run manifests therefore remain in
one workspace instead of silently splitting custom-project state from a
checkpoint written beside the installed source package.

V1 is deliberately not guessed into V2: it never recorded enough information
to prove which random world created its agent state. Attempting to resume V1
fails with a fresh-start instruction. This is a safe compatibility boundary,
not automatic data deletion; the old file remains untouched for inspection.

No training, tournament, formal experiment, future generation or provider call
is executed in this stage. Frozen result identities remain stale for current
code and receive no promotion credit.

## 100. V3.82 Portable run-input identity for resume

V3.81 bound checkpoint progress to the original random root, but equal seeds do
not prove equal worlds. Modified Python sources, changed raw data, a different
world-model checkpoint, altered score/cognition flags or a different runtime
platform could still continue the same checkpoint and produce a false claim of
deterministic resume.

Run manifest V2 adds a portable immutable-input identity. It hashes every Python file below the
installed `src` tree using relative paths and content digests, rather than
trusting Git HEAD or an absolute checkout path. Data and model artifacts retain
relative paths when they belong to the project and content hashes in all cases.
The identity also covers Python/platform, structured `SimulationConfig`, and
all non-secret `MATCH_*` runtime options plus selected model/provider settings.
Credential-like variable names are excluded, and every retained supplemental
value is represented only by its SHA-256 digest. Changes remain detectable
without publishing a value that may have been misused to carry a credential.
Arbitrary `SimulationConfig.extras` follow the same rule: sensitive-looking
keys are omitted and all other values are canonically hashed.

Checkpoint V3 stores the manifest identity under its own content checksum.
Resume rebuilds the current manifest, validates its self-identity and compares
it with the checkpoint before writing a replacement manifest, resolving an LLM
provider or mutating match state. The manager repeats the identity comparison
for direct integrations and refuses full-tournament execution when no verified
identity was supplied. Configured world-model and frozen-shot artifacts are
included when present.

The immutable data set includes all four canonical history CSVs, tactics,
coach profiles, all base roster JSON files and pre-run product-session state
when present. Existing configured `MATCH_*` file inputs are also hashed.
The checkpoint and run manifest themselves are outputs and are intentionally
not self-included.

Squad carryover is different: it is an evolving output and a required resume
input. Each checkpoint therefore stores the current carryover file hash rather
than freezing its initial hash into the run identity. Loading compares the
external file set and content exactly. A crash or external edit that advances
carryover without the matching checkpoint fails closed instead of replaying a
match on top of ambiguous state.

V1 lacks a random-world identity and V2 lacks the full run identity. Neither
can be safely promoted by inference, so both fail with explicit fresh-start
instructions and remain untouched on disk. The checkpoint digest detects
corruption or casual rewriting but is not a cryptographic signature against an
actor able to replace both payload and digest.

No training, tournament, formal experiment, future generation or provider call
is executed in this stage. This strengthens reproducibility mechanics; it does
not create new model-quality, action-effect or outcome evidence.

This stage does not claim complete world-state restoration. Squad carryover is
bound, but all mutable Agent psychology/memory, the social topic market and
writeable cognitive cache directories are not yet one atomic snapshot. Until
that later contract exists, V3 prevents silent input drift and detects known
carryover divergence; it does not authorize an exact-continuation claim for
every optional subsystem.

## 101. V3.83 Dynamic-world recovery and receipted reflection

Checkpoint V4 adds the missing simulation-owned dynamic state. The snapshot
contains every initialized mutable Agent field (latent psychology, affect,
memory layers, beliefs, decision history, governance, physical readiness and
tactical controls), optional match-created tactical/narrative state, dynamic
coach preset adaptations, the social post graph, dialogue topic market and
narrative-event history. Static priors, source data, model services and loaded
world-model objects are deliberately rebuilt from the immutable inputs already
bound by manifest V2 rather than serialized as opaque Python objects.

Snapshot values use a narrow tagged JSON codec for finite scalars, mappings,
lists, tuples, NumPy arrays, `PsychologicalState` and `NarrativeEvent`. Unknown
dataclasses, non-string mapping keys, non-finite floats, malformed tags,
missing core Agent fields or a changed Agent/random/coach identity fail closed.
Reserved codec keys inside ordinary mappings are escaped rather than
reinterpreted, ndarray dtypes are restricted to JSON-safe numeric/boolean
forms, and checkpoint documents are capped at 64 MiB.
Restore performs a complete live-identity and optional-field-presence preflight
before changing the world date, feed or any Agent, preventing a half-restored
in-memory process.

Reflection is now a two-phase simulation transaction. The provider request is
parsed and bounded to 256 KiB without changing Agent state. Its receipt is
checkpointed first; only then is the response applied under a stable operation
ID. The resulting meta-proposal identity is deterministic, the Agent audit
guards against repeat application, and the operation is checkpointed as
applied. If the process stops after the receipt commit but before application,
resume uses the receipt and makes no second provider request. Parse errors are
no longer swallowed as false success. Group-stage operations are keyed by team;
knockout operations are keyed by match and winner. The completed-match skip
path runs any missing winner reflection and reconstructs a final result that
could otherwise be lost between the match checkpoint and the next statement.
Checkpoint validation also cross-checks every receipt, applied marker and
Agent audit: unknown owners, missing audits, premature mutation, duplicate
operation IDs and audit entries without a durable receipt are rejected. Each
receipt is capped at 256 KiB and the journal at 16 MiB.

The causal external-state set now includes squad carryover, persistent
world-model fusion history, tactical counterfactual evidence and, when
cognition is enabled, the complete cognitive cache directory tree. Files are
content-hashed; directory identity hashes relative file names and contents and
rejects symlinks. Their exact set and digest are verified at checkpoint load.
These histories influence later match planning, so treating them as incidental
logs would make resume semantically incorrect.

The real roster audit exposed one pre-existing non-standard `NaN` observation
set in the South Africa data. The source artifact is retained unchanged.
`load_roster_json()` now treats NaN/Infinity as absent observations while
preserving legitimate JSON nulls, and every match ability conversion has a
finite fallback. A team-dynamics vector is attached only when all five causal
components are present and finite; otherwise the established status-derived
vector is used. This prevents both NaN propagation and the misleading partial
vector whose missing attack/defence fields would previously become zeros.

The boundary is crash-consistent for checkpoint-owned in-memory state and
fail-closed for external files. It is not yet an atomic multi-file database:
if carryover, fusion history or a cache file is durably advanced but the next
checkpoint is not, resume refuses the mismatch rather than automatically
rolling the file back. Also, without provider-supported idempotency there is an
unavoidable narrow uncertainty window between receiving a response and
durably saving its receipt; V4 guarantees no repeat call after the receipt is
committed, not universal exactly-once billing. Those are explicit later
operations gates, not hidden claims.

No training, tournament, formal experiment or provider call is executed in
this stage. The real-world snapshot check only constructed and serialized the
333-Agent initial world. Therefore V3.83 improves recovery correctness and data
safety but adds no model-quality, action-effect or outcome evidence.

## 102. V3.84 Recoverable external-state transaction

Checkpoint V5 closes the remaining internal external-state crash gap without
turning resume into an unrestricted filesystem restore. Alongside the digest
map, each checkpoint embeds a versioned rollback image for every present
causal state artifact: squad carryover, world-model fusion history, tactical
counterfactual evidence and the enabled cognitive cache tree. Snapshot files
are base64 transport values with independent SHA-256 identities. Directory
identity is reconstructed from ordered relative paths and file hashes using
the same contract as live-state verification.

The image is deliberately bounded to 32 MiB and 10,000 files, while the full
checkpoint and recovery journal are each bounded to 64 MiB. Absolute paths,
parent traversal, backslash aliases, duplicate paths, malformed base64,
digest disagreement, non-file targets and symlinks fail closed. Missing
artifacts are meaningful: if an artifact was created after the checkpoint,
recovery removes it so the exact checkpoint set is restored.

Resume separates inspection from mutation. Seed extraction and run-manifest
verification load the checkpoint with external comparison disabled, but all
checkpoint checksums, schemas, world state, reflection consistency and
embedded snapshot hashes are still verified. Only after the current
code/data/model/configuration identity equals the checkpoint identity may the
application call external-state recovery. The normal TournamentManager load
then uses strict live comparison and must pass before in-memory restoration.

Recovery holds an operating-system file lease and writes an atomic recovery
journal containing the displaced state before changing the first target.
Individual files are restored by fsync plus atomic replacement. Cache
directories are rebuilt in a confined sibling staging directory. If the
process stops after any target, the journal remains and the next resume
repeats the desired writes until the exact digest set converges. On success,
the journal is retained as the last-recovery record, so the overwritten state
is not silently discarded.

Automatic mutation is limited to paths whose resolved target remains inside
the verified project root. A cognitive cache configured outside that root is
still captured and integrity-checked, but drift produces an explicit error and
is never overwritten. This is a safety boundary, not an exact-recovery claim
for arbitrary external directories. Provider billing also retains the V3.83
limitation before a reflection receipt has been durably committed.

This stage executes no training, production tournament, formal experiment or
provider request. Unit and integration tests exercise bounded synthetic match
fixtures, but they provide no new model-quality or causal-effect evidence.

## 103. V3.85 Recovery continuity and single-load handoff

The V5 recovery journal is now preflighted on every resume branch, including
the branch where live artifact digests already equal the checkpoint. A journal
must be a regular, bounded JSON file; its schema, target checkpoint content
identity, displaced artifact map and complete rollback snapshot must all
validate. The last-recovery destination must also be a regular file when it
exists. Therefore a stale or conflicting interrupted transaction cannot be
silently renamed merely because another process or operator happened to make
the live files match.

The public tournament entry now parses an existing checkpoint once with
external comparison deferred. The same validated object is passed through
root-seed selection, run-manifest comparison, identity-gated external recovery
and TournamentManager restoration. The manager still performs a final strict
live artifact comparison after recovery. This removes repeated parsing and
base object construction for a checkpoint that may legally approach 64 MiB,
while retaining validation at the mutation boundary. Calls that are not
resuming a checkpoint keep the previous TournamentManager method signature.

Regression coverage injects a valid rollback snapshot carrying the wrong
checkpoint identity while the live filesystem is already consistent and
requires failure without journal archival. A public-entry test also counts
checkpoint loads and requires exactly one for a real resume handoff.

No training, production tournament, formal experiment or provider request is
executed in this stage; bounded test fixtures are verification only.

## 104. V3.86 Code-contract versus evidence maturity

The release control plane previously coupled the `target_hash_lock` gate to
the aggregate reproduction-release result. Because that aggregate correctly
fails while the formal paper result is absent, a fully verified target lock
was reported as failed. The same aggregate paper gate was also classified as
a code gate even though its only failing check requires authorized
confirmatory execution. This made `code_ready=false` describe missing external
evidence rather than incomplete code.

The target-lock gate now depends on its own current artifact hashes, the
verified transitive-lock readiness flag and the four target-lock checks:
completeness/hashes, direct dependency inclusion, safe index/portability
contract and artifact agreement. It does not inherit unrelated paper-result
failure.

Code-contract readiness is computed separately from final release readiness.
The paper side is code-ready only when the paper and finalization identities
are current, the finalization protocol passes, and every paper-package check
other than `formal_result_identity_and_replay_pass` is true. The infrastructure
side still requires every designated lock, SBOM, data-license, runtime,
container-image and validation-protocol gate. Both components are exposed in
`code_contract_checks`. Final release readiness continues to require every
gate, including formal results, human studies, independent reviews, credential
closure, container execution and production validation.

The paper-package check names are an exact fail-closed contract. A missing or
unknown check, or a result state other than the single expected pending formal
result, makes code readiness false rather than relying on a vacuously true
aggregate.

The authoritative state after this correction is therefore:
`code_contract_ready_external_gates_open`, with 10 passed and 11 open release
gates. The target lock is passed; the registered paper package remains open
for the missing confirmatory result. Studio CLI now surfaces status,
`code_ready`, the two code-contract checks, `release_ready`, counts and the
next action, matching the existing Web representation.

This is a semantic correction to reporting and gate composition. It does not
promote the world model, manufacture external evidence or change the product
and academic scores.

## 105. V3.87 Outcome-aligned M2 world-model control

M1 proved that a validated pass head could enter the action simplex, but its
runtime score and its formal outcome objective were not the same quantity.
It also enabled the model for both teams in the same match. That design could
measure distribution movement, but it could not cleanly estimate whether one
team benefited from the policy. The historical M1 conclusion therefore
remains immutable: research_only_default_off.

M2 introduces one actor-centred transition utility shared by training,
validation, runtime planning and realized-outcome attribution:

    U = clip(1.0 * delta_goal_difference + 0.35 * delta_territory
             + 0.15 * delta_possession, -2, 2)

Every term is a change from the current state from the original actor's
perspective. Same-state persistence is exactly zero. Home and away advances
have symmetric signs, and absolute possession is no longer rewarded twice.
The trainer adds this value as a member-specific Bayesian-bootstrap auxiliary
loss under the existing curriculum. The grouped holdout report is separated
by pass, cross, shot and hold; an action receives authority only when the
exact target version was trained, at least 96 samples from at least six groups
exist, its MSE beats zero persistence by at least two percent, and its
prediction/target correlation is non-negative.

Action identity is decoded from the executed six-way one-hot vector. The
pass-outcome head may continue to learn an intercepted attempt as a negative
pass outcome, but a defensive `intercept` can no longer count toward M2 pass
policy-utility samples; zero or malformed actions remain `other` rather than
being relabelled as `hold`.

MATCH_WM_OUTCOME_ALIGNED_POLICY=1 selects M2. In this mode the pass and cross
planners call the multi-step predict_policy_utility path used by the
explanation system. They do not silently fall back to the legacy transition
score when the M2 interface or the exact action gate is absent. The legacy M1
path is retained behind the default-off flag so old experiments and product
behavior remain reproducible. Shot authority remains independently governed
by the frozen shot-head gate; M2 policy evidence cannot bypass it.

Every sampled high-level action now retains its adoption record until the next
observation is encoded. The runner then stores realized policy utility,
component deltas, the value predicted at selection time and action-specific
prediction RMSE. Aggregate records survive the bounded 96-record display
window, so full-match mechanism analysis does not depend on retained examples.
The confirmatory mechanism endpoint uses only attribution-eligible influenced
opportunities; the all-opportunity mean remains diagnostic.

The runtime also accepts MATCH_WM_CONTROL_SCOPE=home|away|both|none.
Calibration presets define M0 as none, historical M1 as both, and M2 as a
separate outcome-aligned pipeline. The preregistered M2 study runs each of 120
fixture-seed units as M0, M2-home and M2-away, for 360 matches. Home and away interventions
control only the declared team and are analysed from that team's perspective.
The matched-seed bootstrap samples within fixture while keeping both mirrored
sides in the same cluster.

Promotion requires all preregistered gates: a sealed checkpoint with pass
policy-utility and two-step authority, a non-trivial counterfactual action
change rate, positive realized transition utility, a positive and meaningful
controlled-team micro-xG-margin effect, and non-inferiority on continuous
external calibration loss with the discontinuous failed-metric penalty set to
zero. Goal difference remains secondary even at 120 matched units because
sparse score events are materially noisier than micro-xG. Protocol,
checkpoint, all simulation source files and every registered data/configuration
input are hashed at first execution. Paths are project-relative, the protocol
and checkpoint must remain inside the project root, and resume also requires
the candidate eligibility report to remain identical.

The budget is prospectively variance-based rather than copied from the
inconclusive 30-pair study. Historical paired home-xG-margin differences had
standard deviation 0.3673. A normal approximation requires 106 matched units
for 80 percent power at two-sided alpha 0.05 to detect a meaningful 0.10
xG-margin effect; M2 fixes 120 units. The historical M1 point estimate is not
used as an expected M2 effect.

M2 also freezes its training contract before optimization: 45 epochs, batch
size 128, effective learning rate 0.0003, GRU transition, three independent
dynamics members, 0.25 two-step and policy-utility objectives, 0.20 semantic
event objective, and no ball-log augmentation. The zero-training preflight
verifies the manifest hashes, disjoint train/dev/sealed groups, strict action
support, non-degenerate utility targets, sequential-pair support, curriculum
activation and ensemble size. On the current manifest it passes every gate:
the training split has 9,497 rows across 53 groups and the development split
has 1,793 rows across 10 groups, including 1,723 strict pass actions and 1,783
aligned two-step pairs. It reports but never loads sealed rows for model
selection.

Candidate authorization is stricter than development validation. The
checkpoint must reproduce the exact frozen training configuration, declare
that sealed data was unused, retain the exact manifest binding, pass its
development pass-utility and two-step gates, and then independently pass both
gates on the sealed split. The previous checkpoint passes sealed two-step
state prediction but its sealed pass-utility MSE is 0.01048 versus the zero
persistence MSE of 0.00853 (skill -22.87 percent); it is therefore correctly
rejected even apart from its missing M2 training contract.

This stage completed code and protocol only. It did not train a new
checkpoint or execute the 360-match study. The current default checkpoint is
correctly rejected because it lacks the outcome-aligned policy-utility
training contract, even though its historical two-step gate remains active.
Formal calibration worlds now ignore ambient seeds and persistent squad state;
every result row records whether the runtime loaded, its exact checkpoint
signature, the controlled side and the M2 policy mode. Analysis rejects any
row whose declared arm and runtime identity disagree. It also reconstructs
the exact 120 registered fixture-seed keys for every arm and rejects duplicate,
foreign or missing units both during incremental persistence and before final
analysis. A complete synthetic 360-row acceptance case proves that all gates
can jointly authorize promotion; a positive confidence interval with only a
0.05 xG-margin point effect still fails the frozen 0.10 football-meaning gate.

The product evidence API exposes M2 independently from the immutable M1
result. It reports the protocol, bound checkpoint, eligibility, fixed budget,
progress and gates, but exposes a result or promotion only when the completed
analysis still matches the current protocol, checkpoint, source and input
identity. A stale or partial M2 artifact therefore fails closed and cannot
rewrite the historical `inconclusive_keep_research_only` conclusion.
Consequently M2 remains unavailable to stable product mode until a newly
trained sealed candidate passes the eligibility check and the complete
preregistered study supports promotion.

## 106. V3.88 Identity-bound M2 research control plane

M2 readiness previously existed in three disconnected places: the training
preflight printed to a terminal, checkpoint qualification was only visible
during a runner invocation, and Studio showed a compact result card. That was
scientifically safe but operationally incomplete. An operator could not tell
whether readiness belonged to current code, whether a checkpoint had already
been qualified without starting the study, or which single action was valid.

The zero-training preflight can now persist
`data/evaluation/m2_training_preflight_v1.json` explicitly through
`--out`. The receipt binds the protocol, frozen dataset manifest and every
declared M2 training code dependency. The expensive preflight still verifies
all trace contents at recording time, and training repeats that verification;
ordinary product status requests verify the smaller receipt identity instead
of reparsing the full trace corpus. A current preflight remains readiness
evidence only: it records no optimization, checkpoint or match result.

Checkpoint qualification can likewise persist a standalone
`m2_candidate_eligibility_v1.json` receipt through `--eligibility-out`.
It uses the exact execution identity later required by the formal study and
records both accepted and rejected candidates. A rejected receipt is not a
tuning invitation: after sealed evaluation is opened, the product workflow
requires preserving that rejection and preregistering any genuinely new
candidate.

Studio composes protocol, preflight, candidate receipt, resumable progress and
current analysis into one validated five-stage workflow:

1. freeze the protocol;
2. verify training readiness;
3. qualify one sealed candidate;
4. execute the fixed mirrored study;
5. decide promotion from all preregistered gates.

The projection exposes exactly one next action and distinguishes preflight
drift, candidate identity drift, candidate rejection, authorized execution,
interrupted execution, current non-promotion and current promotion. It never
executes a command. The Web evidence centre renders the stages and exact
operator commands using text-only DOM construction; training and the 360
formal matches still require an explicit command and authorization.

The current recorded preflight is identity-valid and passes all thirteen
checks. Therefore the workflow is at stage 3, `awaiting_candidate`, with
zero of 360 formal matches executed. No training, checkpoint qualification,
formal match, provider request or promotion occurred in this stage.
