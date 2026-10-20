# -*- coding: utf-8 -*-
"""每日 14:40 刷新 QQQ(新浪美股)与 USDCNY(腾讯)日线缓存。

牢国策略(laoguo_signal.py, 14:49)的纳指混合动量依赖这两个序列;
接口失败时 ext_data 内部保留旧缓存, 本脚本退出码非零由 cron 日志可见。
"""
import ext_data


def main():
    qqq = ext_data.fetch_qqq()
    fx = ext_data.fetch_usdcny()
    print("QQQ %d 行(最新 %s) | USDCNY %d 行(最新 %s)" % (len(qqq), qqq[-1][0], len(fx), fx[-1][0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
