# “动量”机器人查询入口

本机“动量 / 轮动 / 动量信号 / 买什么”命令由另一个应用处理：

- 应用：`/root/dingtalk-stock-bot/app.py` 中的 `_handle_momentum_signal()`。
- 服务：`dingtalk-stock-bot.service`（Gunicorn，监听5000端口）。
- 数据：`/root/stock/signals/latest.txt`。
- 共享排版：`/root/stock/push_signal.py` 的 `build_markdown(text, query=True)`。

该应用目录未使用Git；对应的最小接入改动保存在 [dingtalk_momentum.patch](dingtalk_momentum.patch)。接入只替换查询排版函数，每次查询重新加载共享排版，后续调整标的展示不必重复维护两套模板。

查询展示最近保存信号的原日期，保留“以每日14:50正式推送为准”和看板链接；主动推送继续使用原执行提示。两者都突出主策略目标及影子建议标的。

如果重新部署的是旧版机器人应用，可在其目录先检查再应用补丁：

```bash
patch --dry-run -p1 < /root/stock/integrations/dingtalk_momentum.patch
patch -p1 < /root/stock/integrations/dingtalk_momentum.patch
systemctl kill --kill-who=main --signal=HUP dingtalk-stock-bot.service
```

Gunicorn无`--preload`时，HUP让工作进程重新加载入口；先完成语法/渲染检查再重载。线上已应用本补丁，无需重复执行。

可通过本机 `POST http://127.0.0.1:5000/webhook` 验证精确命令“动量”。使用合成发送者且完全省略`sessionWebhook`，该命令同步返回Markdown JSON，不发送群消息。测试时不要使用会调用其他业务或异步发送的命令。
