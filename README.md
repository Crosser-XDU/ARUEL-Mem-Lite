# AUREL-Mem-Lite

AUREL-Mem-Lite 是面向多模态记忆评测的 Add/Search 服务，版本为 `2.0.0-lite`。GitHub 仓库名为 **`ARUEL-Mem-Lite`**；程序名、健康检查和文档中的方法名为 **`AUREL-Mem-Lite`**。

本项目从 [AUREL-Mem](https://github.com/Crosser-XDU/AUREL-Mem) 派生，主要改动是减少 Add 阶段重复的信息抽取。检索算法沿用原版：三路 BM25 与稠密向量召回、加权 RRF、有限关系与会话邻接扩展，以及原有时间和多样性排序。Search 交付原始证据，由调用方或评测端的回答模型作答。

## Lite 改了什么

| Add 输入 | GPT-4o-mini 抽取 | 存储与索引 |
| --- | --- | --- |
| 纯文本，包括简短回复 | 跳过 | 保存完整原文，直接建立文字索引与向量 |
| 图片已有明确、非空的 `Image Caption:` | 跳过对应图片 | 使用输入 Caption 建索引，保留原图；不声称核验过图文一致性 |
| 图片缺少、仅有空 Caption，或对应关系有歧义 | 对需要处理的图片调用原抽取流程 | 保存模型提取字段及原始证据，保留来源映射、格式校验与错误重试 |

多图消息逐图匹配：每张图前紧邻的文字可提供它自己的 Caption；明确匹配的图片可跳过，未覆盖图片继续交给视觉模型。存在歧义时保守回退，不能把一段 Caption 当成所有图片的说明。`Image Caption` 标签支持大小写、空格、下划线、连字符以及 `:` / `：`。

纯文本和完整 Caption 快路径不另外推断 `facts`、`entities` 或 `state_updates`；这些派生数组保持为空。开启向量检索时，所有消息仍调用 Embedding。默认使用 `gpt-4o-mini`、`text-embedding-v4` 和 **1024 维**向量，通过 OpenAI 官方 Python SDK 的兼容接口调用，模型运行在所配置的 API 服务端，无需本地 GPU 或模型权重。

检索代码保持不变，不代表结果一定不变：快路径改变了进入检索器的索引字段和向量输入。尚未通过完整评测或消融实验证明 Lite 与原版质量相同，也不声称提出了新的检索算法。实现细节见 [Lite 方法说明](docs/lite-method.md)。

## Linux / macOS 快速启动

推荐使用 **Python 3.12**，这是发布前测试使用的版本。以下命令在新克隆的仓库目录中执行：

```sh
git clone https://github.com/Crosser-XDU/ARUEL-Mem-Lite.git
cd ARUEL-Mem-Lite
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，填写自己的模型凭证和独立的记忆服务令牌。下列凭证均为占位符：

```dotenv
OPENAI_API_KEY=replace-with-your-model-api-key
OPENAI_API_BASE=https://api.gpt.ge/v1
OPENAI_MODEL=gpt-4o-mini

EMBEDDING_ENABLED=true
EMBEDDING_API_KEY=
EMBEDDING_API_BASE=
EMBEDDING_MODEL=text-embedding-v4
OPENAI_EMBEDDING_DIMENSIONS=1024

MEMORY_API_TOKEN=replace-with-your-own-service-token
MEMORY_DB=data/memory.sqlite3
LLM_CONCURRENCY=4
UPSTREAM_TIMEOUT=60
EXTRACTION_OUTPUT_MODE=auto
RETRIEVAL_LIMIT=32
```

`OPENAI_API_BASE` 是可更换的兼容网关示例，不是 OpenAI 官方端点。`text-embedding-v4` 需要所选 Embedding 服务实际支持；使用官方 SDK 不会自动赋予该模型的访问权限。留空 `EMBEDDING_API_KEY` / `EMBEDDING_API_BASE` 时，分别复用聊天模型的 Key / Base URL；也可以配置独立的向量服务。返回向量必须符合配置维度，失败时不会静默替换模型。

`MEMORY_API_TOKEN` 用于客户端访问本记忆服务，与模型 API Key 分开。请替换占位符并保持 `.env`、数据库及日志在公开仓库之外。配置默认从仓库根目录 `.env` 读取，也可通过 `ENV_FILE` 指定其它文件；已存在的进程环境变量优先。

启动单个应用 worker：

```sh
MEMORY_HOST=127.0.0.1 MEMORY_PORT=18090 \
  PYTHON_BIN="$PWD/.venv/bin/python" bash scripts/run.sh
```

SQLite 默认创建在仓库内的 `data/memory.sqlite3`。更换部署目录时可将 `MEMORY_DB` 设为自己的持久化绝对路径。运行多个独立实例时，应分别配置端口和数据库。

另开终端检查：

```sh
curl -fsS http://127.0.0.1:18090/healthz
```

健康检查表示应用存活及配置是否存在，不证明上游 Key、额度或每个模型调用均可用。Linux 后台服务可参考 [systemd 示例](scripts/fusion-memory.service)；公网 HTTPS 反向代理可参考 [Caddy 示例](deploy/Caddyfile.example)，按实际安装位置修改后使用。上面的快速启动命令适用于 Linux 与 macOS，不依赖特定电脑的启动项。

## Add / Search 接口

本地路径为 `POST /add`、`POST /search` 与 `GET /healthz`。设置服务令牌后，Add/Search 支持 `Authorization: Token <令牌>` 和 `Authorization: Bearer <令牌>`；Health 无需令牌。

在客户端终端中设置自己的服务令牌。以下示例会在本地数据库写入独立的演示记录；开启 Embedding 时会调用已配置的向量 API。

```sh
MEMORY_TOKEN='replace-with-your-own-service-token'

curl -fsS http://127.0.0.1:18090/add \
  -H "Authorization: Token $MEMORY_TOKEN" \
  -H 'Content-Type: application/json' \
  --data '{
    "request_id": "demo-add-001",
    "user_id": "demo-user",
    "session_id": "demo-session",
    "messages": [
      {"role": "user", "timestamp": 1704067200000,
       "content": "我的蓝色望远镜放在书房。"},
      {"role": "assistant", "content": "记住了。"}
    ]
  }'
```

成功返回：

```json
{"success": true, "request_id": "demo-add-001", "user_id": "demo-user", "session_id": "demo-session"}
```

`timestamp` 可省略；提供时使用 Unix 毫秒。`role` 为 `user` 或 `assistant`。同一用户下，相同 `request_id` 和相同内容可幂等重试；复用该 ID 提交不同内容返回 409。Add 在抽取、向量生成和数据库事务全部成功后才返回成功；上游失败不会提交部分消息。

```sh
curl -fsS http://127.0.0.1:18090/search \
  -H "Authorization: Token $MEMORY_TOKEN" \
  -H 'Content-Type: application/json' \
  --data '{"user_id":"demo-user","query":"望远镜放在哪里？","top_k":5}'
```

Search 返回如下结构，实际 ID、内容和分数由检索结果决定：

```json
{
  "data": [
    {
      "id": "record-id",
      "content": "[Evidence ... | role=user | source_time=...]\n我的蓝色望远镜放在书房。",
      "score": 0.03,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

`content` 为文字或有序的多模态 content parts；图像证据会包含原始图片 Data URI。`score` 是排序分数，不是正确率或概率。没有记录时返回 `{"data": []}`。Search 没有 `answer` 字段，不负责选择题选项；可选 `options: ["...", "..."]` 只用于扩展检索文字。纯文本查询只需向量调用；图片查询沿用视觉特征描述流程，再检索证据。

图片输入沿用 OpenAI Chat Completions content parts，例如将消息的 `content` 替换成以下数组（把 `<BASE64>` 替换为实际图片编码）：

```json
[
  {"type": "text", "text": "Image Caption: 一只带白色星形图案的红色杯子。"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>", "detail": "high"}}
]
```

删除 Caption 或仅保留普通文字时，图片进入模型抽取路径。Search 的 `query` 也支持相同的多模态数组。仅接受内联 JPEG、PNG、WebP，不接受远程图片 URL。

## 容量与运行边界

- 单次 Add 最多 1024 条消息，HTTP 请求体最多 48 MiB；单图最多 10 MiB、4000 万像素，不接受动画；一次 Add 或 Search 的输入图片合计最多 30 MiB。
- `top_k` 接收 1–100，默认实际检索上限为 32。默认交付预算为 12 张图片、图片合计 30 MiB、100000 个文本字符；达到预算会标明省略或截断，原始存储不随之删除。
- 每条向量输入沿用最多 8000 字符的限制；完整原文仍保存并用于 BM25。图片向量来自 Caption 或提取文字，不是直接对像素做 CLIP/SigLIP 联合嵌入。
- SQLite 持久化数据，当前 Search 读取该用户索引并在进程内排名。尚未建立大规模吞吐或长期容量保证。减少 GPT 抽取调用不等于固定倍数的整体加速：Embedding、网络、磁盘和调用方发包速度仍会影响耗时。

## 测试与复现

在仓库根目录、激活虚拟环境后执行：

```sh
python -m pytest -q -p no:cacheprovider tests
```

测试使用合成消息和图片、临时数据库以及 HTTP MockTransport，不需要真实模型凭证或评测数据。覆盖 Caption 跳过与局部回退、纯文本直存、来源顺序、官方 SDK 请求序列化、格式兼容与拒绝、向量维数、原图交付、用户隔离、幂等、失败回滚和继承的检索行为。

2026-10-30 的 Python 3.12 测试记录为 **364 passed，48 subtests passed**。该结果证明所测接口与行为，不代表多模态记忆准确率或正式排行榜成绩。项目不借用参考论文的分数，也未声称 Lite 已取得官方评测成绩。

## 已部署接口

| 接口 | URL |
| --- | --- |
| Add | `https://139.196.16.167/lite/add` |
| Search | `https://139.196.16.167/lite/search` |
| Health | `https://139.196.16.167/lite/healthz` |

访问 Add/Search 需要该部署授权的记忆服务令牌；公开仓库不包含它。自托管实例使用你自己配置的令牌和数据库。这些部署地址不是运行开源代码的必要依赖，也不构成长期可用性承诺。

## 研究来源与归属

原版研究了 M2A、V-Mem、ATM-Bench / SGM、GraphMemix、TAME、Omni-SimpleMem 与 DeliverMem。Lite 继承独立编写的工程实现，未打包或运行这些参考仓库，不是七篇论文的完整复现。尤其 TAME 的 GPL 参考源码没有被复制或作为运行依赖。

完整作者、论文、官方代码、锁定研究提交及实现边界见 [研究披露](docs/research-disclosure.md)；Lite 的具体变化见 [方法说明](docs/lite-method.md)。参考仓库的许可证不自动决定本项目的许可证，运行依赖分别适用其自身许可。
