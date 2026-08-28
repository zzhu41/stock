# -*- coding: utf-8 -*-
"""v9 策略研究报告 PDF 生成器 (matplotlib PdfPages, 中文字体 Noto Sans CJK)。

用法: python3.8 make_paper.py  ->  /root/stock/v9_strategy_paper.pdf
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

import backtest
from market_data import UNIVERSE, fetch_history

# ---------- 字体 ----------
for f in ("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc"):
    font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK JP"  # ttc首face; JP/SC共享字形库, 中文显示无碍
plt.rcParams["axes.unicode_minus"] = False

PAGE = (8.27, 11.69)  # A4
MARGIN_L = 0.09

# ---------- 数据 ----------
print("跑 v9 全周期回测...")
histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
res = backtest.backtest(histories, calendar, start="2014-01-01")
daily = res["daily"]
nav300 = backtest.buy_and_hold(histories, calendar, "510300", start="2014-01-01")[0]

# 基准净值序列(与daily同日历)
close300 = {r[0]: r[2] for r in histories["510300"]}
days_all = [d for d, _, _ in daily]
b300_series = []
b0 = close300.get(days_all[0])
for d in days_all:
    b300_series.append(close300.get(d, b300_series[-1] if b300_series else b0))
b300_series = [x / b300_series[0] for x in b300_series]

nav_series = [n for _, n, _ in daily]
dd_series = []
peak = 1.0
for n in nav_series:
    peak = max(peak, n)
    dd_series.append(n / peak - 1)

yearly = backtest.yearly(daily)
# 基准逐年
last300 = {}
for d in days_all:
    last300[d[:4]] = close300.get(d)
b_yearly = []
base = None
for y in sorted(last300):
    v = last300[y]
    if v is None:
        continue
    if base is None:
        b_yearly.append((y, 0.0))
    else:
        b_yearly.append((y, v / base - 1))
    base = v
b_yearly = dict(b_yearly)

# 回撤区间
idx = {d: i for i, (d, _, _) in enumerate(daily)}
eps = []
peak_nav, peak_date = daily[0][1], daily[0][0]
trough_nav, trough_date = daily[0][1], daily[0][0]
in_dd = False
for d, nav, _ in daily:
    if nav >= peak_nav:
        if in_dd:
            eps.append((trough_nav / peak_nav - 1, peak_date, trough_date, d))
            in_dd = False
        peak_nav, peak_date, trough_nav, trough_date = nav, d, nav, d
    else:
        in_dd = True
        if nav < trough_nav:
            trough_nav, trough_date = nav, d
if in_dd:
    eps.append((trough_nav / peak_nav - 1, peak_date, trough_date, None))
eps.sort()

# ablation 数据 (v8消融, 组件贡献, 手工录入自 ablation_v8 输出)
ABLATION = [
    ("牛熊开关", 11.5), ("跨境池", 10.4), ("急跌离场 panic", 4.7),
    ("永不空仓", 4.7), ("绝对动量门", 3.6), ("黄金熊市参赛", 3.0),
    ("WLS25 排名", 1.9), ("滞回缓冲 2%", 1.4),
]

CRASH_BUYS = res["crash_buys"]

# ---------- 排版工具 ----------
def new_page(pdf, title=None, subtitle=None):
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    if title:
        fig.text(MARGIN_L, 0.955, title, fontsize=17, fontweight="bold", color="#1a1a2e")
    if subtitle:
        fig.text(MARGIN_L, 0.928, subtitle, fontsize=10.5, color="#555555")
    return fig


def para(fig, x, y, text, width_chars=88, dy=0.0165, size=9.5, color="#222222", bold=False):
    """简单中文折行排版, 返回新 y。"""
    import textwrap
    lines = []
    for raw in text.split("\n"):
        if not raw.strip():
            lines.append("")
            continue
        buf = ""
        for ch in raw:
            buf += ch
            # 中文按2宽估
            w = sum(2 if ord(c) > 127 else 1 for c in buf)
            if w >= width_chars:
                lines.append(buf)
                buf = ""
        if buf:
            lines.append(buf)
    for ln in lines:
        fig.text(x, y, ln, fontsize=size, color=color,
                 fontweight="bold" if bold else "normal")
        y -= dy
    return y


def table(fig, x, y, headers, rows, col_w=None, row_h=0.019, size=8.8,
          header_color="#1a1a2e", zebra=True):
    """手排表格, 返回新 y。col_w 为相对宽度列表。"""
    ncol = len(headers)
    if col_w is None:
        col_w = [1.0] * ncol
    tot = sum(col_w)
    widths = [w / tot * (0.86) for w in col_w]
    xs = [x]
    for w in widths[:-1]:
        xs.append(xs[-1] + w)
    # header
    fig.patches.append(plt.Rectangle((x - 0.005, y - row_h * 0.75), sum(widths) + 0.01,
                                     row_h * 1.35, transform=fig.transFigure,
                                     facecolor="#eef0f6", edgecolor="none", zorder=0))
    for i, h in enumerate(headers):
        fig.text(xs[i] + 0.004, y, h, fontsize=size, fontweight="bold", color=header_color)
    y -= row_h * 1.5
    for ri, row in enumerate(rows):
        if zebra and ri % 2 == 1:
            fig.patches.append(plt.Rectangle((x - 0.005, y - row_h * 0.72), sum(widths) + 0.01,
                                             row_h * 1.3, transform=fig.transFigure,
                                             facecolor="#f7f8fb", edgecolor="none", zorder=0))
        for i, c in enumerate(row):
            fig.text(xs[i] + 0.004, y, str(c), fontsize=size, color="#222222")
        y -= row_h
    return y - row_h * 0.4


# ---------- PDF ----------
pdf = PdfPages("/root/stock/v9_strategy_paper.pdf")

# ===== 封面/摘要 =====
fig = plt.figure(figsize=PAGE)
fig.patch.set_facecolor("white")
fig.text(0.5, 0.80, "ETF 动量轮动交易系统 v9", fontsize=24, fontweight="bold",
         ha="center", color="#1a1a2e")
fig.text(0.5, 0.765, "体制感知 · 危机增强的单标的动量轮动策略", fontsize=13,
         ha="center", color="#555555")
fig.text(0.5, 0.73, "—— 基于 12.6 年回测与 210+ 变体消融的完整研究 ——", fontsize=10,
         ha="center", color="#888888")
fig.text(0.5, 0.665, "2026 年 8 月", fontsize=11, ha="center", color="#555555")

y = 0.60
fig.text(MARGIN_L, y, "摘  要", fontsize=14, fontweight="bold", color="#1a1a2e")
y -= 0.03
abstract = ("本报告完整描述一套 ETF 动量轮动交易系统（v9）。策略在 11 只 ETF 构成的多资产池上运行，"
            "以牛熊体制开关（沪深300ETF 对年线）做第一层风控，以 25 日时间加权回归斜率（WLS）经波动率调整后排名做标的选择，"
            "配合滞回缓冲、三重离场线（动量转负/单日急跌/过热回落）、熊市进场门槛与永不空仓的避险兜底机制，"
            "并在极端恐慌（5 日跌超 8% 且低于年线 20%）时执行逆向抄底。"
            "2014-01 至 2026-08 回测（含双边万二费用）：年化收益 +49.2%，最大回撤 -27.0%，夏普 1.64，卡玛 1.82，"
            "总收益 +15367%（同期沪深300ETF +206%），13 个自然年无一亏损。"
            "全部规则经 13 轮、210 余个变体的系统性消融与证伪检验，五个存活组件均在多起点、参数邻域与逐年切片上稳健。")
y = para(fig, MARGIN_L, y, abstract)
y -= 0.02
kw = "关键词：动量轮动；趋势跟踪；牛熊体制；危机 Alpha；ETF；绝对收益"
para(fig, MARGIN_L, y, kw, size=9, color="#555555")
pdf.savefig(fig)
plt.close(fig)

# ===== 1 引言 =====
fig = new_page(pdf, "1  引言与相关工作", "为什么又一套动量轮动系统")
y = 0.885
y = para(fig, MARGIN_L, y, (
    "动量效应是金融学中最稳健的异象之一（Jegadeesh & Titman 1993；Moskowitz, Ooi & Pedersen 2012 的时序动量；"
    "Antonacci 的双动量）。但在 A股落地时，教科书式的动量策略会遇到三个本土化问题：\n"
    "（1）政策市特征导致熊市中出现大量 5~15% 的假反弹，动量转正后追入即被埋（熊市陷阱）；\n"
    "（2）单标的满仓轮动的净值曲线波动极大，常规回撤控制手段（移动止损/波动率目标）都会显著侵蚀收益；\n"
    "（3）极端事件（2015 股灾、2020 疫情、2024 微盘崩）中动量信号全面失效。\n"
    "本系统的答案是三层结构：体制层（年线开关）隔离假反弹、动量层（WLS 斜率排名）执行轮动、"
    "危机层（深跌恐慌抄底）把系统性崩塌转化为收益来源。所有组件均由数据证伪驱动筛选："
    "13 轮共 210+ 个候选变体中仅 5 个通过全部稳健性检验（多起点/参数邻域平台/逐年无显著受损）。"))
y -= 0.02
y = para(fig, MARGIN_L, y, (
    "与同类公开实现的对比：某同族平台（3~4 只池：纳指/黄金/创业/国债，2016-07~2026-07）实现年化 40.1%、"
    "回撤 -21.7%、Calmar 1.85。本系统在同区间的年化 42.0%、回撤 -27.0%、Calmar 1.56（v9 口径）。"
    "两者的组件级 ablation 结论高度互证（OLS 口径、FIP、多周期集成、Top-K 持仓在两边均为负优化），"
    "差异主要来自标的池广度（本系统 11 只含商品/红利/小盘）与危机处理方式（对方用空仓冷却，本系统用抄底锁仓）。"), size=9)
pdf.savefig(fig)
plt.close(fig)

# ===== 2 数据 =====
fig = new_page(pdf, "2  数据与回测口径", None)
y = 0.885
y = para(fig, MARGIN_L, y, (
    "行情数据：腾讯财经前复权日 K（含开盘价/收盘价/成交量），本地 CSV 缓存增量更新；"
    "QQQ（新浪）与 USDCNY 汇率（腾讯）用于外部数据研究（最终未采纳进生产规则）。"
    "回测区间 2014-01-01 ~ 2026-08-21（部分标的上市日晚于起点，数据不足期自动不参与排名）。\n"
    "成交假设：信号日收盘价成交（实盘为 14:50 尾盘限价单，滑点敏感性实测单边 0.1% ≈ 年化 -3pp，"
    "故实盘要求限价贴价）；费用：双边万二（佣金万一 ×2）；空仓期停泊货币 ETF（511880 真实净值）。"))
y -= 0.01
y = para(fig, MARGIN_L, y, "标的池（11 只）", bold=True)
rows = [
    ("159915", "创业板ETF", "A股竞赛池", "成长进攻"),
    ("588080", "科创50ETF", "A股竞赛池", "科技进攻"),
    ("510300", "沪深300ETF", "A股竞赛池", "大盘核心 / 牛熊基准"),
    ("510500", "中证500ETF", "A股竞赛池", "中盘"),
    ("563300", "中证2000ETF", "A股竞赛池", "小盘"),
    ("512400", "有色金属ETF", "A股竞赛池", "商品周期"),
    ("512890", "红利低波ETF", "A股竞赛池", "防御价值"),
    ("513100", "纳指ETF", "跨境池(不受开关约束)", "美股科技"),
    ("513120", "港股创新药ETF", "跨境池(不受开关约束)", "港股医药"),
    ("518880", "黄金ETF", "备胎/熊市参赛/避险兜底", "危机对冲"),
    ("511880", "货币ETF", "停靠", "空仓利息"),
]
y = table(fig, MARGIN_L, y, ["代码", "名称", "角色", "定位"], rows, col_w=[1, 1.6, 2.2, 1.4])
pdf.savefig(fig)
plt.close(fig)

# ===== 3 策略规则 =====
fig = new_page(pdf, "3  策略规则（v9 完整规范）", "七组件：体制 → 排名 → 缓冲 → 离场 → 兜底 → 熊市门槛 → 危机抄底")
y = 0.885
rules = [
    ("3.1 牛熊体制开关", "沪深300ETF 收盘价 > MA250 年线为牛市，反之为熊市。熊市中 A股池 7 只无条件禁入"
     "（实测：2022-06 创业板假反弹 MOM20 高达 +21.8%，熊市假反弹的动量读数为全年最强，任何动量门槛均无法过滤，"
     "只能靠体制层一刀切）。熊市竞赛池 = 跨境 2 只 + 黄金。"),
    ("3.2 排名因子（WLS25）", "score = Slope25_W ÷ Vol20。对近 25 日收盘价做时间加权最小二乘回归（权重线性递增，"
     "近期更高），斜率年化后除以 20 日收益波动率。较首尾比值法（MOM20/VOL20）使用全路径信息、抗噪、换手更低（201 vs 251 次）。"
     "窗口 23~28 为连续平台，时间加权经消融验证为必要成分。"),
    ("3.3 进场与轮动缓冲", "第一名且 MOM20>0（牛市）/ MOM20>7%（熊市，过滤弱反弹）方可进场；"
     "挑战者 MOM20 须超现持仓 2 个百分点才触发轮动（滞回缓冲，防噪声横跳）。"),
    ("3.4 三重离场线", "(a) MOM20 转负（趋势死亡）；(b) 单日跌幅 ≤ -4%（急跌保险，次日若仍第一可立即买回，无冷却）；"
     "(c) 过热态（MOM20>+40%）下 MOM5 转负（抛物线顶快离场）。"),
    ("3.5 永不空仓兜底", "竞赛池无人达标时，依次取：牛市黄金备胎（MOM20>0）→ 避险池（纳指/创新药/黄金）score 最强者。"
     "不蹲货币干等（避险资产与 A股低相关，全灭日往往恰是其启动时）。"),
    ("3.6 危机抄底（危机 Alpha）", "任何标的 MOM5 ≤ -8% 且价格低于年线 20%（深跌恐慌）时，当日抄底买入并锁仓 5 个交易日"
     "（期内不响应任何其他信号）。深跌约束把 2015 顶部崩盘（不抄）与 2015-08/2022/2024 深跌恐慌（抄）精确区分。"
     "12.6 年仅触发 15 次，9 个年份完全静默。"),
]
for title, body in rules:
    y = para(fig, MARGIN_L, y, title, bold=True, size=10.5)
    y = para(fig, MARGIN_L + 0.012, y, body, size=9)
    y -= 0.012
pdf.savefig(fig)
plt.close(fig)

# ===== 4 回测结果: 净值图 + 回撤图 =====
fig = plt.figure(figsize=PAGE)
fig.patch.set_facecolor("white")
fig.text(MARGIN_L, 0.955, "4  回测结果", fontsize=17, fontweight="bold", color="#1a1a2e")
fig.text(MARGIN_L, 0.928, "2014-01 ~ 2026-08，含双边万二费用", fontsize=10.5, color="#555555")

ax1 = fig.add_axes([0.10, 0.56, 0.84, 0.32])
ax1.plot(range(len(nav_series)), nav_series, lw=1.4, color="#c0392b", label="策略 v9（对数轴）")
ax1.plot(range(len(b300_series)), b300_series, lw=1.1, color="#7f8c8d", label="沪深300ETF")
ax1.set_yscale("log")
ax1.set_ylabel("净值（对数）")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(alpha=0.25)
ax1.set_title("累计净值曲线", fontsize=11)
xticks = list(range(0, len(days_all), 488))
ax1.set_xticks(xticks)
ax1.set_xticklabels([days_all[i][:4] for i in xticks], fontsize=8)

ax2 = fig.add_axes([0.10, 0.30, 0.84, 0.20])
ax2.fill_between(range(len(dd_series)), [x * 100 for x in dd_series], 0,
                 color="#c0392b", alpha=0.35, lw=0.5)
ax2.set_ylabel("回撤 (%)")
ax2.grid(alpha=0.25)
ax2.set_title("回撤曲线（最大 -27.0%，2020-02 新冠流动性危机）", fontsize=11)
ax2.set_xticks(xticks)
ax2.set_xticklabels([days_all[i][:4] for i in xticks], fontsize=8)

txt = ("总收益 +15367%（1 元 → 154.7 元）｜年化 +49.2%｜最大回撤 -27.0%｜夏普 1.64｜卡玛 1.82｜"
       "年换手约 18 次｜13 个自然年无一亏损")
fig.text(0.5, 0.24, txt, fontsize=10, ha="center", color="#1a1a2e", fontweight="bold")
pdf.savefig(fig)
plt.close(fig)

# ===== 4.1 逐年表 + 柱状图 =====
fig = plt.figure(figsize=PAGE)
fig.patch.set_facecolor("white")
fig.text(MARGIN_L, 0.955, "4.1  逐年收益", fontsize=15, fontweight="bold", color="#1a1a2e")
years = [y for y, _ in yearly]
v9y = [x * 100 for _, x in yearly]
b300y = [b_yearly.get(y, 0) * 100 for y in years]

ax = fig.add_axes([0.10, 0.52, 0.84, 0.36])
xx = np.arange(len(years))
ax.bar(xx - 0.19, v9y, width=0.38, color="#c0392b", label="策略 v9")
ax.bar(xx + 0.19, b300y, width=0.38, color="#95a5a6", label="沪深300ETF")
ax.set_xticks(xx)
ax.set_xticklabels(years, fontsize=8.5)
ax.axhline(0, color="#333", lw=0.6)
ax.legend(fontsize=9)
ax.grid(alpha=0.25, axis="y")
ax.set_ylabel("年度收益 (%)")
for i, v in enumerate(v9y):
    ax.text(i - 0.19, v + (2 if v >= 0 else -5), "%.0f" % v, ha="center", fontsize=7.2, color="#7b241c")

rows = [(y, "%+.1f%%" % v, "%+.1f%%" % b, "%+.1fpp" % (v - b)) for y, v, b in zip(years, v9y, b300y)]
table(fig, MARGIN_L, 0.44, ["年份", "策略 v9", "沪深300ETF", "超额"], rows,
      col_w=[1, 1.2, 1.4, 1.1], row_h=0.0175, size=8.6)
pdf.savefig(fig)
plt.close(fig)

# ===== 5 归因 =====
fig = plt.figure(figsize=PAGE)
fig.patch.set_facecolor("white")
fig.text(MARGIN_L, 0.955, "5  收益来源归因", fontsize=17, fontweight="bold", color="#1a1a2e")
fig.text(MARGIN_L, 0.928, "组件消融（逐个拆除后的年化损失）+ 危机抄底逐笔审计", fontsize=10.5, color="#555555")

ax = fig.add_axes([0.32, 0.60, 0.62, 0.30])
names = [x[0] for x in ABLATION][::-1]
vals = [x[1] for x in ABLATION][::-1]
ax.barh(range(len(vals)), vals, color="#2c3e50")
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("拆除该组件的年化损失（pp）")
ax.grid(alpha=0.25, axis="x")
ax.set_title("组件贡献（v8 基线消融；v9 抄底另计）", fontsize=10.5)
for i, v in enumerate(vals):
    ax.text(v + 0.15, i, "%.1f" % v, va="center", fontsize=8.5)

y = 0.55
y = para(fig, MARGIN_L, y, (
    "结构结论：收益的最大支柱是体制层（牛熊开关 +11.5pp）与资产广度（跨境池 +10.4pp），而非排名细节——"
    "方向正确比精度重要。三代升级（panic +4.7 / 永不空仓 +4.7 / WLS +1.9pp）各自独立有效；"
    "11 个组件中 9 个有正贡献（过热与黄金备胎为零贡献的沉睡保险），不存在过拟合系统典型的"
    "「拆掉某组件反而更好」现象。"), size=9)
y -= 0.015
y = para(fig, MARGIN_L, y, "危机抄底逐笔（节选）：", bold=True, size=9.5)
cb_rows = []
close_of = {c: {r[0]: r[2] for r in rows_} for c, rows_ in histories.items()}
cal_idx = {d: i for i, d in enumerate(calendar)}
for d, c in CRASH_BUYS[:10]:
    p0 = close_of[c].get(d)
    j = cal_idx.get(d)
    f10 = close_of[c].get(calendar[j + 10]) if j and j + 10 < len(calendar) else None
    cb_rows.append((d, UNIVERSE[c][0], "%+.1f%%" % ((f10 / p0 - 1) * 100) if f10 else "-"))
y = table(fig, MARGIN_L, y, ["抄底日期", "标的", "其后10日"], cb_rows,
          col_w=[1.3, 1.6, 1.1], row_h=0.0175, size=8.6)
pdf.savefig(fig)
plt.close(fig)

# ===== 5.1 危机年表现 =====
fig = new_page(pdf, "5.1  危机年份的净值行为", "回撤区间解剖")
y = 0.885
dd_rows = []
causes = {"2020-02-13": "新冠流动性危机", "2021-11-22": "全球齐跌前夜", "2020-09-03": "纳指回调",
          "2026-01-29": "有色闪崩(panic拦截)", "2026-05-25": "震荡打脸期", "2024-07-09": "日元套息平仓"}
for depth, p, t, rec in eps[:6]:
    rec_s = rec if rec else "未修复"
    cause = causes.get(p, "")
    dd_rows.append(("%.1f%%" % (depth * 100), "%s~%s" % (p[2:], t[2:]), rec_s[2:] if rec else "-", cause))
y = table(fig, MARGIN_L, y, ["深度", "区间(峰→谷)", "修复日", "病因"], dd_rows,
          col_w=[0.9, 1.9, 1.0, 2.2], row_h=0.019, size=8.6)
y -= 0.015
y = para(fig, MARGIN_L, y, (
    "回撤结构特征：最深回撤 -27.0% 发生于 2020 年 2-3 月全球流动性危机（美元挤兑下黄金与风险资产同跌，"
    "无差别抛售无对冲可躲），73 个交易日修复；其次为 2021-11 起的 -22.0%（全球齐跌，修复 220 天，为最长套牢期）。"
    "8 段主要回撤中位数修复时间 21 天——危机层的抄底模块显著压缩了尾部修复时长。"
    "净值约 89% 的时间处于回撤/爬坡状态，为趋势策略常态。"), size=9)
pdf.savefig(fig)
plt.close(fig)

# ===== 6 稳健性 =====
fig = new_page(pdf, "6  稳健性与证伪史", "210+ 变体的系统性检验")
y = 0.885
y = para(fig, MARGIN_L, y, (
    "采纳标准（全部满足才可转正）：(1) 6 个回测起点（2014/2016/2018/2020/2022/2024）终值全部改善；"
    "(2) 参数邻域为连续平台而非孤峰（防过拟合）；(3) 逐年切片无显著受损年份；(4) 全周期含费复现。"))
y -= 0.01
y = para(fig, MARGIN_L, y, "存活组件（5/210+）：", bold=True, size=9.5)
y = table(fig, MARGIN_L, y, ["组件", "引入版本", "核心证据"], [
    ("急跌离场 panic 4%", "v6", "6/6起点全赢；11次触发6次躲开续崩"),
    ("永不空仓（避险池兜底）", "v7", "6/6起点终值全赢；2018年+9.4pp"),
    ("WLS25 斜率排名", "v8", "6/6全赢；窗口23-28平台；换手-20%"),
    ("熊市进场门槛 MOM20>7%", "v8.1", "6/6全赢；门槛5-10%平台"),
    ("深跌恐慌抄底", "v9", "6/6全赢；9年静默；回撤不变反浅"),
], col_w=[2.0, 1.0, 3.0], row_h=0.019, size=8.6)
y -= 0.015
y = para(fig, MARGIN_L, y, (
    "证伪清单（节选，均经全周期回测否决）：移动止损、熔断冷却、绝对止损、空仓冷却、确认天数、多周期动量集成、"
    "52周高点、跳过近期动量（短期反转过滤）、FIP 信息离散度、MAX 彩票效应、Sortino 下行波动排名、"
    "波动率目标仓位（有效但牺牲终点，用户目标函数下否决）、Top-2 分散持仓（终值腰斩）、"
    "MACD/KDJ/RSI/均线/布林/CCI 全部 36 个技术指标变体、唐奇安低点止损、动量衰减止损、"
    "QQQ 混合动量（7 档）、成交量三维（量比门槛/天量禁追/量比加权）、ER 震荡自适应缓冲、同因子防切、"
    "t 统计量排名、周频信号、熊市开放 A股（所有门槛）。\n"
    "注：MA200 牛熊开关曾在单区间诱人（+2.2pp），但起点敏感性 3 好 2 差、逐年 3 年大损，为典型过拟合形态，否决。"), size=8.8)
pdf.savefig(fig)
plt.close(fig)

# ===== 7 风险与局限 =====
fig = new_page(pdf, "7  风险与局限", None)
y = 0.885
y = para(fig, MARGIN_L, y, (
    "(1) 回测≠未来：全部历史业绩基于 2014-2026 单一市场 regime，动量溢价在未来若结构性减弱（策略拥挤化），"
    "收益中枢将下移。\n"
    "(2) 单标的满仓：集中度风险始终存在，-27% 级别回撤是系统性风险税，无法在不牺牲终点的前提下消除"
    "（实测：高波禁进场可降回撤至 -21%，但 2025-26 高波主升浪终点受损更大）。\n"
    "(3) 逐年波动：WLS 排名在结构震荡市（如 2021/2023 型年份）会显著跑输其前身（-17~-19pp），"
    "执行心理成本需预期管理。\n"
    "(4) 执行摩擦：回测以收盘价成交，实盘 14:50 限价单存在滑点（单边 0.1%≈年化-3pp），须严格贴价；"
    "QDII 标的（513100/513120）买入需检查溢价率（>2% 不追）。\n"
    "(5) 抄底模块的低频性：12.6 年 15 次触发，统计样本小；其收益集中在 4 次历史级危机，"
    "下一次危机形态未必同构。\n"
    "(6) 数据依赖：腾讯/新浪免费接口的稳定性与复权口径未经审计，生产环境需每日校验信号文件时间戳。"), size=9)
pdf.savefig(fig)
plt.close(fig)

# ===== 8 结论 =====
fig = new_page(pdf, "8  结论", None)
y = 0.885
y = para(fig, MARGIN_L, y, (
    "本报告展示了一套经 13 轮、210+ 变体证伪驱动收敛的 ETF 动量轮动系统。其超额收益的来源可分解为："
    "体制层的假反弹隔离（牛熊开关）、资产广度的全天候轮动（11 只跨类别池）、趋势精度的抗噪提升（WLS 斜率）、"
    "尾部风险的双向利用（急跌离场躲连续崩盘 + 深跌抄底接恐慌极点）、以及熊市假反弹的弱信号过滤（7% 门槛）。\n"
    "最终形态 2014-2026 年化 +49.2%、最大回撤 -27.0%、夏普 1.64、13 年无亏损自然年。"
    "策略已进入「只维护、不开发」阶段：任何进一步的参数改动都必须先通过同等的四维稳健性检验，"
    "否则视为过拟合拒绝。"), size=9.5)
y -= 0.03
y = para(fig, MARGIN_L, y, (
    "附录：代码与数据管道（/root/stock）：market_data.py（腾讯前复权+增量缓存）、strategy.py（全部规则与试验开关）、"
    "backtest.py（回测引擎，信号/收益分离）、signal_daily.py（14:50 每日信号+抄底锁仓持久化）、"
    "push_signal.py（钉钉群推送）、record.py（实盘账本）。复盘命令：python3.8 backtest.py 2014-01-01。"), size=8.5, color="#555555")
pdf.savefig(fig)
plt.close(fig)

pdf.close()
import os
sz = os.path.getsize("/root/stock/v9_strategy_paper.pdf")
print("PDF 生成完成: /root/stock/v9_strategy_paper.pdf (%.0f KB)" % (sz / 1024))
