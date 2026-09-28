# “动量”机器人查询入口

本机“动量 / 轮动 / 动量信号 / 买什么”命令由另一个应用处理：

- 应用：`/root/dingtalk-stock-bot/app.py` 中的 `_handle_momentum_signal()`。
- 服务：`dingtalk-stock-bot.service`（Gunicorn，监听5000端口）。
- 数据：优先读取`signals/daily_state.json`权威记录；旧记录兼容`signals/latest.txt`。
- 共享排版：`/root/stock/push_signal.py` 的 `render_saved_query(path=...)`（内部共用 `build_markdown`）。

该应用目录未使用Git；对应的最小接入改动保存在 [dingtalk_momentum.patch](dingtalk_momentum.patch)。接入只替换查询排版函数，每次查询重新加载共享排版，后续调整标的展示不必重复维护两套模板。

查询展示最近保存信号的原日期，保留“以每日14:50正式推送为准”和看板链接；主动推送继续使用原执行提示。两者都突出主策略目标及影子建议标的。

当前首卡为用户指定的 **V12-R2主推送（R2提高年化）**，随后为V9.2、V9.2+、V10-H对照。本入口自动展示新主的加粗目标、日期及等待/失败状态，无需修改外部应用。查询不会计算策略、初始化账户或推进净值；缺主信号不复制对照目标，历史卡明确标注非当前操作指令。

如果重新部署的是旧版机器人应用，可在其目录先检查再应用补丁：

```bash
patch --dry-run -p1 < /root/stock/integrations/dingtalk_momentum.patch
patch -p1 < /root/stock/integrations/dingtalk_momentum.patch
systemctl kill --kill-who=main --signal=HUP dingtalk-stock-bot.service
```

Gunicorn无`--preload`时，HUP让工作进程重新加载入口；先完成语法/渲染检查再重载。线上已应用本补丁，无需重复执行。

可通过本机 `POST http://127.0.0.1:5000/webhook` 验证精确命令“动量”。使用合成发送者、合法的本机回调token header且完全省略`sessionWebhook`，该命令同步返回Markdown JSON，不发送群消息。未认证请求现在返回401；不得把token打印到终端或日志。测试时不要使用会调用其他业务或异步发送的命令。

2026-09-28补充了实际会话回复链路：**已认证**的动量回调携带合法、未过期的官方sessionWebhook时，向该次会话显式回发Markdown，HTTP仅返回ACK；无会话地址时保留旧同步响应。地址限制为官方HTTPS `oapi.dingtalk.com/robot/sendBySession`，禁止重定向，不会回退发送到固定群。只有回发HTTP成功且整数errcode=0才记录sent，收到/排队/HTTP ACK都不等于已送达。进程内缓存抑制同msgId重试，但不承诺跨重启或多worker严格一次。

接入补丁为[dingtalk_auth_presence.patch](dingtalk_auth_presence.patch)与[dingtalk_momentum_reply.patch](dingtalk_momentum_reply.patch)，按各自review JSON中的before/after SHA核对源再应用。它们基于已经完成安全接入的应用，不要对未知基线盲目重复打补丁。回调认证模式保持显式token/sign，不能仅凭本机配置存在token就认定真实机器人协议正确；拒绝日志只记录请求头存在性和来源是否loopback，不记录任何凭据值或正文。

回调鉴权、日志脱敏和无关配置隔离的追加补丁见 [安全部署说明](dingtalk_security.md)。定时链路与状态恢复见 [运维说明](daily_system.md)。当前运行版本已应用两份补丁，不要重复应用旧排版补丁。
