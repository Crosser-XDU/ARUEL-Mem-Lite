# AUREL-Mem C

本分支 `codex/v3-c-mlx-reranker` 对应 **C：B 的公共逻辑 + Qwen3-Reranker-4B MLX MXFP8 重排**。不下载或启用 SigLIP2。

已部署地址：`POST https://139.196.16.167/m3/add`、`POST https://139.196.16.167/m3/search`。服务 Token 私下配置；仓库仅含占位符。

## 启动

推荐 Python 3.12。C 的模型 worker 需要 Apple Silicon macOS / MLX。

```sh
git clone --branch codex/v3-c-mlx-reranker --single-branch https://github.com/Crosser-XDU/ARUEL-Mem-Lite.git
cd ARUEL-Mem-Lite
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-core.lock
python -m pip install --no-deps -e .
cp .env.example .env
chmod 600 .env
```

编辑 `.env`，填写服务 Token、上游模型 Key；Embedding 服务必须支持 `text-embedding-v4`、1024 维。未单独设置 Embedding 地址/Key 时复用聊天模型地址/Key。示例中的兼容网关地址可以替换。

在 `.env` 中另设 `WORKER_TOKEN`（API 与 worker 共用该内部 Token）及 `MODEL_ROOT`。创建独立模型环境并下载固定权重：

```sh
python3.12 -m venv .venv-models
.venv-models/bin/python -m pip install -r requirements-models.lock
.venv-models/bin/python scripts/download_reranker.py
.venv-models/bin/python scripts/run.py worker
```

另一个终端运行 API。worker 初次装载完毕后，API 的 `/healthz` 才返回 200。`requirements-models.lock` 是已验证环境的完整快照，含未启用的 Torch 依赖；实际 C worker 不加载 SigLIP2。

```sh
.venv/bin/python scripts/run.py api
# 默认监听 127.0.0.1:18120
```

`ENV_FILE=/absolute/private.env` 可指定配置；`--host` / `--port` 可调整 API 监听。worker 强制回环地址。上线可通过 LaunchAgent/systemd 托管单进程 API 及 C 的单例 worker，避免多个 worker 重复加载模型。

## 接口与方法

- [API 契约与示例](docs/api.md)：与赛事 [官方指南](https://agentmemoryleaderboard.ai/api-guide) 对齐，输出原始证据，不输出最终问答答案。
- [结构提取、维护、检索和输出预算](docs/method.md)：最多三轮检索、两次充分性判断，原图最多 12 张，补充最多 30 组 Caption/上下文。
- [部署结构](docs/deployment.md)与 [Caddy 示例](deploy/Caddyfile.example)。
- [方法来源](docs/research-disclosure.md)与[验证记录](docs/validation.json)。

GPT-4o-mini 使用 JSON object 模式，格式问题本地修复/归一化/来源回退，不因格式错误重复请求。网络失败有独立、有限的传输重试。Add 幂等、持久化后回执；不同 user_id 数据隔离。

## 测试

```sh
.venv/bin/python -m pytest -q
```

原 Lite 版本保留在 `codex/v2.0-lite` 分支。本分支更新默认服务和测试入口，旧 Lite 数据不自动迁移；本次线上 B/C 数据使用其已有独立目录。
