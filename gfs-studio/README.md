# GFS Studio

`gfs-studio` 在 GFS Core 之上提供研究证据、世界模型、训练、认知角色、经理工作流、参与者研究和 loopback Web。比赛内核由 Core 提供，不在本项目中复制。

## 安装

需要 Python 3.11–3.13，以及与当前 Studio 兼容的 Core：

```bash
python -m pip install -e ./gfs-core -e ./gfs-studio
```

普通 Web、备份、证据和离线规则认知不需要 Torch。训练与世界模型使用可选 ML 依赖：

```bash
python -m pip install -e "./gfs-studio[ml]"
```

## 使用

```bash
gfs-studio --base-dir ./gfs-studio research status
gfs-studio --base-dir ./gfs-studio release status
gfs-studio --base-dir ./gfs-studio web
```

打开 <http://127.0.0.1:8765>。Web 仅绑定本机，不提供远程生产认证。

比赛通过任务执行：

```bash
curl -X POST http://127.0.0.1:8765/api/micro-match \
  -H "Content-Type: application/json" \
  -d '{"home":"Brazil","away":"Argentina","seed":42}'
```

响应为 HTTP `202`，包含 `task_id` 和 `status_url`。查询该地址直到状态变为 `review`、`failed`、`interrupted` 或 `cancelled`。服务启动时恢复中断任务，关闭时在 Core tick 边界协作取消运行任务。

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/api/tasks
curl -X POST http://127.0.0.1:8765/api/tasks/cancel \
  -H "Content-Type: application/json" -d '{"task_id":"TASK_ID"}'
```

## 插件与证据

Studio 沿用 `contracts → kernel → plugins → bootstrap → api`，通过 `gfs_studio.plugins` entry point 支持受信任的第三方扩展。能力注册必须与 manifest 一致。

证据身份分为 Core、Studio、验证器、部署和仿真范围；无关验证脚本变化不会使仿真证据失效。`code_ready` 只表示测试、干净 wheel 安装和当前构建检查通过。`release_ready` 还需要真实运行、研究协议、独立复现和参与者证据。

## 容器

先构建版本化 Core 基础镜像，再构建 Studio：

```bash
docker build -t gfs-core:7.0.0a1 ./gfs-core
docker compose -f deployment/compose.yaml build studio
docker compose -f deployment/compose.yaml up studio
```

Compose 只启动 Studio，Core 作为其内部库。锁文件使用哈希安装，Python 基础镜像固定 digest。

## 验证

```bash
python -m pytest tests -q
uvx --from ruff==0.15.17 ruff check src tests
```

工作区根目录的 `python scripts/verify_release.py` 同时验证两个项目、供应链、wheel 干净安装和静态部署。完整外部研究门禁保持独立，不能由单元测试替代。
