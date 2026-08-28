#!/bin/bash
# 每日 23:30 自动快照: 有变更才提交
cd /root/stock || exit 1
git add -A
git diff --cached --quiet && exit 0
git commit -q -m "auto: $(date +%F) 日常快照"
