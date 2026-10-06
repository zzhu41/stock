#!/bin/bash
# 每日 23:30 自动快照: 有变更才提交, 提交后推送 GitHub
cd /root/stock || exit 1
git add -A
git diff --cached --quiet && exit 0
git commit -q -m "auto: $(date +%F) 日常快照"
# 直连优先, 失败回退 xray 代理(GitHub 直连在本机不稳定)
git push -q origin main 2>/dev/null \
  || git -c http.proxy=http://127.0.0.1:7890 push -q origin main 2>/dev/null \
  || echo "git push 失败(直连与代理均不通), 保留本地提交明日重试"
