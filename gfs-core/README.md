# GFS Core

`gfs-core` 是足球社会仿真的独立运行包：球队数据、宏观/微观比赛、锦标赛、情绪动力学、长期记忆和 checkpoint。它不依赖 Torch、LLM 或 Web 框架。

## 安装

需要 Python 3.11–3.13。在包含 `gfs-core` 与 `gfs-studio` 的工作区中：

```bash
python -m venv .venv
python -m pip install -e ./gfs-core
```

Linux 部署使用带完整哈希的 `requirements-linux-py312.lock`；`supply-chain/sbom.cdx.json` 描述对应运行时闭包。

## 运行

仓库自带的示例数据通过 `--base-dir` 指定；单独安装 wheel 时，需要另外提供合法的数据工作区。

```bash
gfs-core --base-dir ./gfs-core status --top 10
gfs-core --base-dir ./gfs-core micro --home Brazil --away Argentina --seed 42 --fast --json
gfs-core --base-dir ./gfs-core monte-carlo -n 100 --seed 42 --json
```

`--fast` 是流程演示配置，不是完整时长校准证据。微观比分模式分为 `scheduled`、`physics` 和 `hybrid`，结果保留来源。

```python
from gfs_core.bootstrap.application import CoreApplication

app = CoreApplication("gfs-core", workspace_root="tmp/core-workspace")
summary = app.run_micro_match("Brazil", "Argentina", seed=42, fast=True)
print(summary.goals_micro_home, summary.goals_micro_away)
```

## 插件结构

```text
contracts/       稳定的数据与生命周期契约
kernel/          注册、拓扑解析、能力、事件、持久化
plugins/         数据、比赛、锦标赛、记忆、展示等官方插件
bootstrap/       按场景组装依赖
api.py           公共入口
```

插件同时具有实现与 `plugin.toml`。内核检查依赖环、API/合同版本、声明能力、激活回滚和逆序关闭。第三方包可通过 `gfs_core.plugins` entry point 注册；Python 插件属于受信任代码，不是安全沙箱。

记忆使用 SQLite 不可变事件和比赛级事务。容量指标不会删除历史事件，历史 `as_of` 回放不随当前容量变化。旧 JSON 存储会校验并迁移一次。

## 验证

```bash
python -m pytest tests -q
uvx --from ruff==0.15.17 ruff check src tests
```

完整 CI 还会构建 wheel、在干净环境安装并执行真实微观比赛。代码通过不代表足球或社会结论已获外部验证。
