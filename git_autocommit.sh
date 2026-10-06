#!/bin/bash
# 每日 23:30 自动快照: 有变更才提交, 提交后推送 GitHub
cd /root/stock || exit 1
git add -A
git diff --cached --quiet && exit 0
git commit -q -m "auto: $(date +%F) 日常快照"
git push -q origin main || echo "git push 失败(网络/认证), 保留本地提交明日重试"
