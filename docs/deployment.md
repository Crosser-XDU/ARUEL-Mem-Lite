# 当前部署

本分支对应版本 B。公网 `POST https://139.196.16.167/add`、`POST https://139.196.16.167/search`，`GET https://139.196.16.167/healthz`。
鉴权 Token 由部署者通过私有环境变量配置，不写入本仓库。

运行端为 Apple Silicon Mac；167 服务器的 Caddy 终止 TLS 并经过双 SSH 反向隧道访问本机。

| 版本 | Mac 监听 | 167 隧道端口 | 入口 |
| --- | --- | --- | --- |
| B | 127.0.0.1:18110 | 127.0.0.1:18111、18112 | /add、/search |
| C | 127.0.0.1:18120 | 127.0.0.1:18121、18122 | /m3/add、/m3/search |

本机 LaunchAgents 名称：`ai.aurel.memory.abc.b`、`.c`、`.worker`、`.tunnel1`、`.tunnel2`。
服务与隧道 KeepAlive、自启，Caddy 通过 /healthz 检查后端。模型 worker 只监听 127.0.0.1:18220，单例串行执行。
Mac 必须保持运行、登录会话和网络可用。仓库不包含机器专用私有 env、LaunchAgent 绝对路径或 SSH 凭据。

当前入口保留 /memory-b 与 /memory-c 兼容别名，/lite 服务路由不变。/stats 只供本机携带服务 Token 查看，公网不开放。

新机器使用 README 的启动命令和 deploy/Caddyfile.example；示例不能直接覆盖现有多服务 Caddyfile。备份、校验配置后做 graceful reload。B、C 使用独立数据目录；工作线程 Token 与 API Token 分开设置。
