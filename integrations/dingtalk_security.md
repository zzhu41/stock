# 钉钉查询入口的鉴权与故障隔离

本模块保护外部 `/root/dingtalk-stock-bot` 的消息回调。`dingtalk_guard.py` 为纯函数，不读环境、不写文件、不发送消息；与 Python 3.6 兼容。实际接入补丁为 `dingtalk_security.patch`，只修改 `app.py` 和 `dingtalk.py` 的相关函数。

协议依据：阿里官方 [chatbot.install 参数说明](https://developer.alibaba.com/docs/api.htm?apiId=47514) 定义 `outgoing_token` 通过回调的 `token` 请求头提供；企业应用机器人另有 [HTTP 接收消息协议](https://open.dingtalk.com/document/orgapp/receive-message)，使用 `timestamp` / `sign` 与应用 AppSecret。两种协议不能混用。

本次现场检查只记录存在性：运行服务 `DINGTALK_OUTGOING_TOKEN` 已配置；回调 AppSecret 和显式模式未配置。因此默认使用旧版 Outgoing `token` 模式，恒定时间比较；缺配置返回 503，凭据缺失或错误返回 401。没有使用向群发消息的 `DINGTALK_WEBHOOK_SECRET` 替代回调凭据。

可选显式企业模式：`DINGTALK_CALLBACK_AUTH_MODE=sign`，配合 `DINGTALK_CALLBACK_APP_SECRET`。该模式验证原始 Base64 签名及正负一小时窗口；计算式为 `Base64(HMAC-SHA256(AppSecret, timestamp + "\n" + AppSecret))`。它不会在失败时尝试 token 模式，也不接受 URL 查询参数中的发群签名。现场没有部署这种模式，其协议仅做离线合成验签测试。

回调在解析正文前鉴权。正文最多读取 128 KiB 加一个探测字节，超限返回 413；非法 JSON、非对象或错误的文本/路由字段类型返回 400。日志只记录固定认证结果或异常类别，不保存原始正文、用户文本、Token、签名或 `sessionWebhook`。

六个消息发送辅助函数的错误日志也已收敛：只记录异常类名或整数 `errcode`，不记录可能带凭证 URL 的 `requests` 异常文本，也不记录远端完整错误响应。此处只改变日志内容，不改变外发流程、超时或成功判定。

“动量”分支先于无关股票列表读取，避免坏股票配置阻断保存信号查询。其它命令遇到配置读取失败会返回明确提示。查询统一调用 `push_signal.render_saved_query(path=...)`，由共享只读加载器处理已保存任务、日期、生成和推送失败，不自行重算策略或推进账户。

部署前先保留外部两个文件的私有备份；完整外部源码不要提交进本仓库。`dingtalk_security_review.json` 记录应用前后源码 SHA256。已有共享排版接入的现场可先 `patch --dry-run -p1` 检查本补丁，再由部署方应用。若从更旧机器人恢复，需要先检查并应用原 `dingtalk_momentum.patch`，再应用本补丁。共享 guard 和 `render_saved_query` 必须先就绪，然后才统一重载服务。

离线回归：

```bash
python3.8 -B -m unittest discover -s tests -p test_dingtalk_security.py -v
```

测试覆盖缺配置拒绝、正确/错误 token、正确/错误/过期/未来签名、禁止模式降级、正文大小及类型、未认证不能进入路由、日志不含敏感正文、坏股票配置不阻断动量、共享查询接口。外部应用只抽取相关函数到隔离测试命名空间；不启动其调度器、不调用外部网络、不发送消息。

部署方还应在本机内存中用现场配置的 token 做一次合法请求验证、一次错误 token 验证；不得打印凭据或把它放入命令行参数。认证只解决应用回调来源检查，不代表已经审计公网入口、HTTPS终止或所有其它机器人业务。

2026-09-27 已在本机部署并重载。内存使用现场token的本机“动量”请求返回200，未认证401、非法正文400、超限413；未提供sessionWebhook，未外发群测试消息。备份及核验状态见review JSON的deployment字段。
