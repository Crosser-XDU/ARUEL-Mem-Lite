# 方法来源与验证边界

本版本在 Lite 之后重新实现结构化记忆和检索流程。MemCell、MemScene、事件持续维护和充分性判断的设计参考 EverMemOS / EverOS 及 MemBase 中的 EverMemOS 基线；图片与来源对话的联合交付参考 Vanilla RAG Memory。这里是面向当前 API 的简化实现，不是直接部署完整 EverMemOS，也不声称复现其排行榜成绩。

- [MemBase 的 MobileMem-Omni 评测与基线](https://github.com/zjunlp/MemBase/tree/a9ba9495d48541d3e68588ff097294cc7a1201bc/examples/evaluate_memory_systems_on_mobilemem_omni)
- [EverOS](https://github.com/EverMind-AI/EverOS)
- [Vanilla RAG Memory](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory)
- [Qwen 官方 Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-4B)

研究时读取的文件、固定 revision 与 SHA256 见 research-manifest.json。模型权重不包含在仓库中；模型来源和量化格式见根目录 model-manifest.json。

39 项自动化测试及真实模型、公网链路验证见 validation.json；发布分支另行运行同一套测试。这些不等于官方比赛分数，尚未重新跑完整榜单评测。
