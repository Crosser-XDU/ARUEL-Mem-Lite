# Add / Search API



```text
Authorization: Bearer <token>
Authorization: Token <token>
X-Api-Key: <token>
```

Add 示例：

```json
{
  "request_id": "request-001",
  "user_id": "user-001",
  "session_id": "session-001",
  "messages": [
    {"role": "user", "content": "这个红色背包属于我妹妹小雨。", "timestamp": 1790812800000}
  ]
}
```

成功时原样返回三个 ID，以及 `"success": true`。成功响应代表消息、图片出现记录、结构数据及检索向量均已持久化并可立即搜索。同一用户重复提交同一 `request_id`、相同内容直接返回原回执，不再次调用模型；同 ID 不同内容返回 409。

Search 示例：

```json
{"user_id":"user-001","query":"红色背包是谁的？","top_k":100}
```

响应为 `{"data":[{"id":"...","content":"...","score":1.0}]}`。带图片的 content 是有序 text/image_url 数组，图片为 Base64 Data URI。无结果返回 `{"data":[]}`。不会返回自写的最终问答答案。

消息和 query 均支持纯文本或官方多模态数组；`options` 是可选字符串列表。只接受内联 JPEG/PNG/WebP，每张解码后不超过 10 MiB，Add/查询图片总量不超过 30 MiB；公网和本机请求体上限一致为 48 MiB。Search 返回解码后的图片总量不超过 30 MiB。`top_k` 支持 1–100，用户数据按 `user_id` 隔离，可跨 session 检索。

契约来源：[赛事官方 API 指南](https://agentmemoryleaderboard.ai/api-guide)。
