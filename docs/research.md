# 多模态记忆系统：源码研究与迁移记录

这是原版 AUREL-Mem 的历史源码研究记录（2026-10-28，Asia/Shanghai）。官方仓库仅用于只读研究；锁定提交见 [research-manifest.json](research-manifest.json)。Lite 的当前行为以 [lite-method.md](lite-method.md) 为准。

## 方法及对应源码

| 方法 | 锁定 commit | 关键源码位置（相对各自仓库） | 可移植设计 |
|---|---|---|---|
| M2A | `edd8c3b75bae8b2c9c1a0ac8ed67e38c2c2723f8` | `agent/stores/raw.py:18` `RawMessageStore`；`agent/stores/semantic.py:15` `SemanticMemory`；`:264` `hybrid_search`；`:362` `_rrf` | 原始消息层与语义层分离；语义记录保留 `evidence_ids`；RRF 合并不同分数空间 |
| V-Mem | `36916b14dc5241e04acbf5cd0f3c40799bc09550` | `model.py:17` `Unit`、`:49` `Round`；`retrieve.py:79` `retrieve`；`load_memgallery.py:60` `load_rounds` | 文本、caption 和图像各为来源可追踪单元；召回单元后恢复整个轮次及有限邻居；按查询模态路由 |
| ATM-Bench / SGM | `ef4e5dff1a47ec71213a06e359f02753defa8fb1` | `memqa/mem_processor/image/config.py:34` prompts；`image_processor.py:489` `extract_entities`；`:543` `process_single_image` | 将时间、地点、描述、OCR、标签、实体分别编码为字段，形成可检索的多个视角 |
| GraphMemix | `f938a6cebc634d7d8da8971d25d155903c0a52eb` | `scripts/graphmemix_core.py:83` `candidate_pool`；`scripts/build_graphmemix_relation_store.py:70` `explicit_edges`；`scripts/probe_graphmemix_pcsf.py:163` `solve_forest_greedy` | 保留种子召回后做有界关系扩展；区分直接证据与互补证据；在证据预算内减少冗余 |
| TAME | `21d5a7cfa85106f278868e2f60887e8ab264c1f2` | `method/TAME.py:166` `read_history_single_turn`；`method/utils/memory_utils.py:53`、`:117` 静态/动态更新；`:188` 去重提升；`:208` FIFO | 区分稳定事实与短期状态；基于新事件更新当前视图；保留时间依据 |
| Omni-SimpleMem | `db80b6a7c591e0ea730a058e9f5fc4eb06572299` | `OmniSimpleMem/omni_memory/retrieval/pyramid_retriever.py:114` `retrieve_preview`、`:223` `expand`、`:338` `retrieve_with_budget`；`core/config.py:105` `LLMConfig` | 索引读取精简表示，交付时取回原始证据；明确输出预算；模型配置统一 |
| DeliverMem | 无可下载代码 | [官方项目页](https://avalon-s.github.io/DeliverMem/)仍注明 Code on publication；[论文 §5](https://arxiv.org/html/2609.32590v1) | 返回原始图像、可读来源 ID 与出现时间；绑定同条记录的文字和图像 |

## 深读后确认的细节和边界

M2A 的原始库采用 SQLite 顺序消息，语义库用 Milvus；`SemanticMemory.evidence_ids` 保存消息 ID 区间。当前 `hybrid_search` 会创建文字至图片的 `request_3`，但 `text_reqs` 只包含前两个请求，因此不能把当前源码直接称为完整三路调用。原始及语义 Store 的默认 `reuse/re_use=False` 会删除旧数据库；这是研究程序的运行默认值，不应带入长期服务。原始记录只有一个 `image_path`，AML 有序多个 content parts 需要独立适配。

V-Mem 先对单元做 dense/BM25/RRF，再按 `round_id` 取该轮最高单元分数。它会在同一 session 内合并重叠或相邻窗口，块按相关性、块内按时间排序；默认邻居扩展可使返回轮次数超过 top_k。服务必须最后实施协议上限并控制总图片字节。`load_memgallery.py:73` 读取数据集预置的第一条 `image_caption`，不是免费视觉识别；本系统若调用 GPT-4o-mini 自行产生视觉字段，应计入写入成本。原仓库 MiniLM/SigLIP 召回与文字化视觉字段并不等价，不能继承其论文分数。

SGM 当前处理链会分别读取 EXIF/GPS、OCR，调用 caption、short_caption、tags、entities 等函数。合并为一次 GPT-4o-mini JSON 抽取是本任务的成本控制实现；未提供的时间/地点必须留空，不能根据图片猜成事实。源码 `process_single_image` 保存了 timestamp、caption、tags、entities、ocr_text 等独立字段。对于 AML，以官方消息时间为主要来源，并保留“消息时间”与“事件发生时间”的区别。

GraphMemix 的 `candidate_pool` 先锁定 Atomic top-L 种子，再按关系置信度与种子排名有界扩展。`explicit_edges` 使用同轮与相邻会话记录关系；另有时空和互近邻语义边。完整方法有独立 node verifier 与 anchor-conditioned ECV；固定 K 求解器有 MILP 路径和确定性的 1-swap 近似路径。仅实现实体邻居扩展＋覆盖奖励＋重复惩罚，属于受其启发的轻量选择器，不能声称复现 evidence-forest 优化器。`concrete_memix.py:48` 的另一路完整 harness 依赖外置 `vendor/memix-core/scripts/QA_Agent/MMRAG/memix_memory.py`，克隆目录没有该外置包，因此没有将仓库当作可直接运行的服务。

TAME 实际将动态事实提升到静态列表，去除重复，再用容量 10 的 FIFO 限制动态记忆；`modify/remove` 会变更列表。为支持“以前/现在”的 AML 查询，本任务建议保留不可变原始证据，历史状态版本化，当前状态独立计算。这是有意改造，非 TAME 原样机制。原实现还依赖 Grounding DINO、Jina embeddings，与本任务的模型限制不同。

Omni-SimpleMem 的 pyramid retriever 把可检索摘要与原始数据指针分开，按预算展开信息。当前 `core/config.py:115` 的 caption_model 默认仍是 `gpt-4o`，embedding 默认 `text-embedding-3-small`；`:225` 有 `set_unified_model`。当前 entity extractor 已优先采用配置模型并回落到 `gpt-4o-mini`，不能用论文时期的模型说明替代当前源码审计。`pyramid_retriever.py:432` 对输出日期有 `year < 2026` 的基准特定条件，不适用于本服务。参数记忆、自演化、音视频处理不需要纳入首版。

DeliverMem 的原模态交付、来源标识和时间定位可独立实现。论文还包含区域多向量召回及像素标识控制，本任务未复现这些实验；返回原图也不能补回从未进入候选集的视觉线索。

## 独立实现的推荐接口流程

Add：保存有序原始消息及图片 → 单次 GPT-4o-mini 结构化抽取 → 建立原文、事实、视觉/OCR 的 lexical 与指定 embedding 索引 → 事务提交 → 成功回应。原始图片文件可按 hash 去重，但每次出现必须保留独立来源记录。

Search：查询图文解析 → lexical+dense+视角召回 → RRF → 同轮/同实体/有限相邻记录扩展 → 互补覆盖与重复惩罚排序 → 按条数、文本和图片字节预算交付原始图文、来源及时间。Search 只返回证据，不生成最终问题答案。所有数据库查找、关系扩展和缓存必须按 user_id 隔离。

这是组合研究思想后的独立工程实现。没有训练、没有运行论文基准、没有继承论文成绩；只有真实接口与端到端测试能证明服务可运行，效果优劣仍需固定数据的消融实验。

## 许可证与来源

- M2A、V-Mem、ATM-Bench、SimpleMem：根目录 LICENSE 为 MIT。下载的原仓库保留原许可；本任务不直接合并其代码。
- TAME：GPL-3.0。作为只读研究参考，独立表达状态管理思路。
- GraphMemix：`pyproject.toml:10` 声明 MIT，但根目录没有 LICENSE 文件。准确记录这一差异，不把缺少独立许可文件写成完全未声明许可证。
- DeliverMem：没有公布可下载实现。论文与项目页是研究来源，不是已部署代码组件。

论文原始来源：[M2A](https://arxiv.org/abs/2602.07624)、[V-Mem](https://arxiv.org/abs/2608.01543)、[ATM-Bench/SGM](https://arxiv.org/abs/2603.01990)、[GraphMemix](https://arxiv.org/abs/2608.26983)、[TAME](https://arxiv.org/abs/2512.21616)、[Omni-SimpleMem](https://arxiv.org/abs/2604.01007)、[DeliverMem](https://arxiv.org/abs/2609.32590)。

官方仓库：[M2A](https://github.com/Little-Fridge/M2A)、[V-Mem](https://github.com/Dingyi-Kang/V-Mem)、[ATM-Bench](https://github.com/JingbiaoMei/ATM-Bench)、[GraphMemix](https://github.com/memorax-ai/graphmemix)、[TAME](https://github.com/ronpay/TAME)、[SimpleMem](https://github.com/aiming-lab/SimpleMem)。
