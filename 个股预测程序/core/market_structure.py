# -*- coding: utf-8 -*-
"""
A股大盘的三层嵌套分析 —— 大周期牛熊 / 中级行情波段 / 短线结构。

为什么需要这个模块
------------------
个股预测单看个股是不够的。A股有极强的**系统性特征**：
大盘定方向、板块定弹性、个股定alpha。同样一只票，在牛市主升段和熊市阴跌段，
同样的技术形态会有完全相反的结局。

而「看大盘「最容易犯的错是**只看涨跌、不分层级**：
看到指数涨 1% 就说「多头来了」，却不知道这 1% 是大周期底部刚启动（可持续两年），
还是主升段末端的一次反抽（几天就结束）。

所以本模块严格按**三层嵌套**来做：

=================  ==========================================  ============
层级                判据                                       时间尺度
=================  ==========================================  ============
**最上层：大周期**   从低点涨>20%=技术性牛市；跌>20%=技术性熊市   3~5 年
                    A股特征：牛短熊长、暴涨暴跌、政策底领先
**中间层：中级行情** 牛初/牛中/牛末/熊初/熊中/熊末 六波段        数月至一年
**最细层：短线结构** 高点低点抬高=上升；降低=下降；否则震荡       日/周
=================  ==========================================  ============

关键设计
--------
1. **严格因果**：所有指标只用当日及之前的数据，绝无未来函数（有测试保证）。
2. **量价验证**：按用户给的「真假波段识别器「实现——
   低位放量不跌=建仓、涨放量调缩量=健康主升、高位放量不涨=派发、
   价新高量跟不上=顶背离、下跌放量=抛压未尽、缩量阴跌=没见底、
   地量后放量长阳=可能见底。
3. **操作纪律**：不同波段给不同的操作提示（不是一句「看多看空」），
   因为上升波段该低吸、下降波段该等企稳、震荡波段该轻仓。
4. **A股特性**：政策市、资金市、情绪放大、板块轮动有顺序、均值回归强。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .contracts import Category, DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Factor, safe_float
from .utils import get_logger

log = get_logger()

#: 大周期状态
REGIME_BULL = "牛市"
REGIME_BEAR = "熊市"
REGIME_RANGE = "震荡市"

#: 中级行情六波段 —— 用户给定的标准剧本
PHASES = {
    "bull_start": dict(
        label="牛初（修复段）", dir=DIR_BULL,
        traits="政策转向宽松，估值从地板往上抬；券商、超跌权重打头阵，"
               "资金试探性进场；情绪还停留在熊市思维（涨了也不敢信）",
        signals="站上年线、量能温和放大、不破前低",
        play="分批建仓，优先券商与超跌低估值权重；不要等「确认牛市」才动手，"
             "那往往已是牛中",
    ),
    "bull_mid": dict(
        label="牛中（主升段）", dir=DIR_BULL,
        traits="盈利兑现 + 产业主线清晰；均线多头排列、板块轮动有序；"
               "赚钱效应扩散，散户开始进场",
        signals="量价齐升、缩量回踩不破位",
        play="最赚钱的阶段：**持股为主**，缩量回踩关键均线可低吸；"
             "主线板块不要因短期波动下车",
    ),
    "bull_end": dict(
        label="牛末（冲顶段）", dir=DIR_BEAR,
        traits="垃圾股补涨、散户疯狂开户、天量成交；估值泡沫化，"
               "利好当利空都跌不下来",
        signals="高位放量滞涨、量价背离",
        play="**减仓兑现**，不追高、不加杠杆；保留现金等下一轮。"
             "这个阶段赚指数不赚钱，个股风险最大",
    ),
    "bear_start": dict(
        label="熊初（出清段）", dir=DIR_BEAR,
        traits="政策收紧或流动性转向，资金集中撤离；跌破关键支撑，"
               "出现千股跌停式杀跌",
        signals="放量大跌、反弹无量",
        play="**果断降低仓位**，不抄底、不摊平；第一次暴跌后的反弹多是诱多",
    ),
    "bear_mid": dict(
        label="熊中（阴跌段）", dir=DIR_BEAR,
        traits="业绩下修、流动性收紧、情绪绝望；缩量阴跌，"
               "任何反弹都是下跌中继",
        signals="高点不断降低、低点不断创新低",
        play="空仓或极低仓位；如需配置只考虑高股息与公用事业防御；"
             "**不接缩量反弹的飞刀**",
    ),
    "bear_end": dict(
        label="熊末（筑底段）", dir=DIR_NEUTRAL,
        traits="估值跌回历史低位，成交极度萎缩；利空出尽，长线资金悄悄布局",
        signals="地量地价 → 放量长阳 → 回踩不破前低",
        play="**开始关注**，分批试探性建仓；等「放量长阳 + 回踩不破前低」再加重仓位",
    ),
}

#: A股五条独有规律（用于报告与界面说明）
A_SHARE_TRAITS = [
    ("政策市属性", "降准降息、产业扶持、监管松紧直接定方向。政策转向时技术破位也可能是假破位；"
                 "政策收紧时技术好看也扛不住。**A股不是经济晴雨表，是政策+流动性晴雨表**。"),
    ("资金市属性", "水涨船高、水落船低。流动性宽松→估值抬升，收紧→估值回落。"
                 "看两融、ETF 流入、北向资金比看 GDP 更管用。"),
    ("情绪放大", "散户占比高、羊群效应强。同样的利好，A股反应幅度常是美股的 2-3 倍，"
               "涨跌都过度 —— 这既是风险也是超额收益的来源。"),
    ("板块轮动有顺序", "牛初：金融、地产、基建（稳增长）→ 牛中：科技、新能源、高端制造（成长主线）"
                   "→ 牛末：消费、医药补涨（收尾）→ 熊市：高股息、公用事业防御。"
                   "**轮动顺序本身就是判断阶段的线索**。"),
    ("均值回归极强", "涨多了必跌、跌多了必弹。没有永远的高估也没有永远的低估，"
                   "**PE/PB 分位比「故事」更可靠**。"),
]

#: 量价验证口诀（用户给定，作为因子与界面说明使用）
VP_RULES = [
    ("低位放量不跌", DIR_BULL, "有资金建仓"),
    ("上涨放量+回踩缩量", DIR_BULL, "健康主升，可跟"),
    ("高位放量不涨", DIR_BEAR, "有人在派发"),
    ("价创新高但量跟不上", DIR_BEAR, "顶背离，警惕"),
    ("下跌放量", DIR_BEAR, "抛压未尽，别抄"),
    ("缩量阴跌", DIR_BEAR, "没见底，只是流动性枯竭"),
    ("地量后放量长阳", DIR_BULL, "可能见底信号"),
]


# --------------------------------------------------------------------------- #
# 一、最上层：大周期牛熊
# --------------------------------------------------------------------------- #
def detect_regime(df: pd.DataFrame, lookback: int = 1200) -> dict:
    """判定当前处于大周期牛市 / 熊市 / 震荡，并给出关键位置。

    技术定义（朴素且通用）：从前期低点涨超 20% = 技术性牛市；
    从前期高点跌超 20% = 技术性熊市。

    Returns
    -------
    dict：``regime / low_price / high_price / from_low_pct / from_high_pct /
    ma_state / duration_bars / detail``
    """
    out = {"regime": REGIME_RANGE, "detail": "", "direction": DIR_NEUTRAL,
           "strength": 0.3, "reliability": 0.6}
    if df is None or not len(df) or "close" not in df.columns:
        out["detail"] = "缺少指数数据，无法判定大周期"
        return out
    d = df.dropna(subset=["close"]).copy()
    c = d["close"].astype(float)
    n = len(c)
    if n < 60:
        out["detail"] = "样本不足 60 根，无法判定大周期"
        return out
    win = min(n, lookback)
    seg = c.tail(win)
    lo_i, hi_i = int(np.argmin(seg.to_numpy())), int(np.argmax(seg.to_numpy()))
    lo, hi = float(seg.iloc[lo_i]), float(seg.iloc[hi_i])
    cur = float(c.iloc[-1])
    from_low = (cur / lo - 1) * 100 if lo else 0.0
    from_high = (cur / hi - 1) * 100 if hi else 0.0
    # 20% 阈值判定（含「当前离极值的远近「与「均线状态「两个辅助条件）
    ma250 = float(c.tail(250).mean()) if n >= 250 else float(c.mean())
    above_ma = cur >= ma250
    if from_low >= 20 and from_high > -20:
        regime = REGIME_BULL
        direction, strength = DIR_BULL, min(0.45 + from_low / 100.0, 0.85)
    elif from_high <= -20 and from_low < 20:
        regime = REGIME_BEAR
        direction, strength = DIR_BEAR, min(0.45 + abs(from_high) / 100.0, 0.85)
    elif from_low >= 20 and from_high <= -20:
        # 既涨了 20% 又跌了 20% → 宽幅震荡（大区间来回）
        regime = REGIME_RANGE
        direction, strength = DIR_NEUTRAL, 0.35
    else:
        regime = REGIME_RANGE
        direction, strength = DIR_NEUTRAL, 0.3
    # ---- 趋势证据闸门（AI 核对图4 第2条）----
    # AI 指出：单凭"距5年低点 +36.6%"判"牛市"逻辑薄弱——距离低点远
    # 只说明位置高，不等于趋势向上（可能是长期横盘或修复到一半）。
    # 牛市应有趋势证据：站上年线 或 均线多头排列 至少满足其一。
    # 不满足时降级为"长期低位修复（趋势未确认）"，并下调强度。
    _regime_ctx = regime
    if regime == REGIME_BULL and not above_ma:
        # AI 核对指出：距5年低点+36.6%、距高点仅-13%，现价其实在历史区间
        # 偏上位置；叫"低位修复"会误导（低位是相对哪个基准？）。
        # 改为不含位置判断的表述，只描述"趋势证据不足"这一事实。
        regime = "趋势未确认的修复反弹"
        direction = DIR_NEUTRAL
        strength = max(0.20, strength * 0.55)
        _regime_ctx = ("原判「牛市」仅依据距低点 %.1f%%，但价格在年线下方，"
                       "趋势证据不足 → 降级为「趋势未确认的修复反弹」"
                       "（注：距高点 %.1f%%，现价在历史区间**偏上**位置，"
                       "不宜称「低位」）" % (from_low, from_high))
    out["regime_ctx"] = _regime_ctx
    # 持续时间：区间内单调方向的最长连续段（用月线级别的粗略度量）
    out.update({
        "regime": regime, "direction": direction, "strength": round(strength, 3),
        "reliability": 0.72, "current": round(cur, 2),
        "low_price": round(lo, 2), "high_price": round(hi, 2),
        "from_low_pct": round(from_low, 2), "from_high_pct": round(from_high, 2),
        "ma250": round(ma250, 2), "above_ma250": bool(above_ma),
        "window_bars": int(win),
        "window": "%s ~ %s" % (str(d["date"].iloc[-win])[:10],
                               str(d["date"].iloc[-1])[:10]),
    })
    out["detail"] = (
        "大周期判定为「%s」：近 %d 根K线区间 %s（低点 %.2f → 高点 %.2f），"
        "当前 %.2f，距低点 %+.1f%%、距高点 %+.1f%%，%s年线（MA250 %.2f）。"
        % (regime, win, out["window"], lo, hi, cur, from_low, from_high,
           "站上" if above_ma else "跌破", ma250))
    return out


# --------------------------------------------------------------------------- #
# 二、最细层：短线结构（高点低点抬高/降低）
# --------------------------------------------------------------------------- #
def swing_structure(df: pd.DataFrame, window: int = 60, order: int = 3) -> dict:
    """判定短线波段结构：上升 / 下降 / 震荡（用户口诀：高点上移+低点上移=多头）。

    用 ZigZag 思路提取局部极值：一个点是「高点「当它比前后 order 根都高。
    """
    out = {"structure": "震荡", "detail": "", "direction": DIR_NEUTRAL,
           "strength": 0.2, "reliability": 0.55}
    if df is None or not len(df) or "high" not in df.columns:
        out["detail"] = "缺少数据，无法判定短线结构"
        return out
    d = df.tail(max(window * 2, 120)).reset_index(drop=True)
    hi = d["high"].astype(float).to_numpy()
    lo = d["low"].astype(float).to_numpy()
    n = len(hi)
    if n < order * 4:
        out["detail"] = "样本不足，无法判定短线结构"
        return out
    highs_idx, lows_idx = [], []
    for i in range(order, n - order):
        if hi[i] == max(hi[i - order:i + order + 1]):
            highs_idx.append(i)
        if lo[i] == min(lo[i - order:i + order + 1]):
            lows_idx.append(i)
    out["n_highs"] = len(highs_idx)
    out["n_lows"] = len(lows_idx)
    if len(highs_idx) < 2 or len(lows_idx) < 2:
        out["detail"] = "近期极值点太少，结构不明（按震荡处理）"
        return out
    # 只看最近若干段
    h = [hi[i] for i in highs_idx[-3:]]
    l = [lo[i] for i in lows_idx[-3:]]
    h_up = all(h[i] < h[i + 1] for i in range(len(h) - 1)) if len(h) >= 2 else False
    h_dn = all(h[i] > h[i + 1] for i in range(len(h) - 1)) if len(h) >= 2 else False
    l_up = all(l[i] < l[i + 1] for i in range(len(l) - 1)) if len(l) >= 2 else False
    l_dn = all(l[i] > l[i + 1] for i in range(len(l) - 1)) if len(l) >= 2 else False
    if h_up and l_up:
        struct, direction = "上升波段", DIR_BULL
    elif h_dn and l_dn:
        struct, direction = "下降波段", DIR_BEAR
    else:
        struct, direction = "震荡波段", DIR_NEUTRAL
    # 强度：最新高低点相对前一段的幅度
    amp = 0.0
    try:
        if len(h) >= 2 and h[-2]:
            amp = max(amp, abs(h[-1] / h[-2] - 1) * 100)
        if len(l) >= 2 and l[-2]:
            amp = max(amp, abs(l[-1] / l[-2] - 1) * 100)
    except Exception:  # noqa: BLE001
        pass
    out.update({
        "structure": struct, "direction": direction,
        "strength": round(min(0.3 + amp / 10.0, 0.8), 3),
        "reliability": 0.6,
        "recent_highs": [round(x, 2) for x in h],
        "recent_lows": [round(x, 2) for x in l],
        "high_trend": ("上移" if h_up else ("下移" if h_dn else "走平")),
        "low_trend": ("上移" if l_up else ("下移" if l_dn else "走平")),
        "amplitude_pct": round(amp, 2),
    })
    out["detail"] = (
        "短线结构为「%s」：近期高点%s（%s）、低点%s（%s）。"
        "口诀：高点上移+低点上移是多头，反之下移是空头，都不够就是震荡。"
        % (struct, out["high_trend"], "、".join("%.2f" % x for x in h),
           out["low_trend"], "、".join("%.2f" % x for x in l)))
    return out


# --------------------------------------------------------------------------- #
# 三、量价验证（真假波段识别器）
# --------------------------------------------------------------------------- #
def volume_price_check(df: pd.DataFrame, window: int = 20) -> dict:
    """按用户给定的量价口诀判断当前处于哪种量价状态。

    Returns
    -------
    dict：``state / direction / signals(list) / detail``
    """
    out = {"state": "中性", "direction": DIR_NEUTRAL, "signals": [],
           "detail": "", "strength": 0.2}
    if df is None or not len(df) or "volume" not in df.columns:
        out["detail"] = "缺少量能数据，量价验证跳过"
        return out
    d = df.dropna(subset=["close", "volume"]).tail(max(window * 4, 120))
    if len(d) < 30:
        out["detail"] = "样本不足，量价验证跳过"
        return out
    c = d["close"].astype(float)
    v = d["volume"].astype(float)
    n = len(c)
    cur = float(c.iloc[-1])
    ret1 = (cur / float(c.iloc[-2]) - 1) * 100
    ret5 = (cur / float(c.iloc[-6]) - 1) * 100 if n > 6 else 0.0
    vr = float(v.tail(5).mean() / v.tail(20).mean()) if v.tail(20).mean() else 1.0
    win = min(n, 250)
    hi, lo = float(c.tail(win).max()), float(c.tail(win).min())
    pos = (cur - lo) / max(hi - lo, 1e-9)
    low_pos, high_pos = pos < 0.35, pos > 0.70
    big_vol, small_vol = vr > 1.35, vr < 0.75
    # 地量：近 20 日量能是近 120 日的低位
    dpct = 1.0
    if n >= 120:
        base = float(v.tail(120).quantile(0.2))
        dpct = float(v.tail(5).mean() / base) if base else 1.0
    vol_dry = dpct < 1.15
    long_yang = ret1 > 1.5 and big_vol
    # 价创新高但量跟不上（顶背离）
    recent_high = float(c.tail(5).max())
    prev_high = float(c.tail(60).max()) if n >= 60 else recent_high
    new_high = recent_high >= prev_high * 0.999
    v_ma5, v_ma20 = float(v.tail(5).mean()), float(v.tail(20).mean())

    sigs = []
    direction = DIR_NEUTRAL
    strength = 0.2
    state = "中性"

    if low_pos and big_vol and ret5 > -1.0:
        state, direction = "低位放量不跌", DIR_BULL
        sigs.append("低位放量不跌 → 有资金建仓")
        strength = 0.62
    elif ret5 > 1.5 and big_vol:
        state, direction = "上涨放量+回踩缩量", DIR_BULL
        sigs.append("上涨放量 → 健康主升，可跟")
        strength = 0.66
    elif high_pos and ret5 < 1.0 and big_vol:
        state, direction = "高位放量不涨", DIR_BEAR
        sigs.append("高位放量不涨 → 有人在派发")
        strength = 0.68
    elif new_high and v_ma5 < v_ma20 * 0.9:
        state, direction = "价创新高但量跟不上", DIR_BEAR
        sigs.append("价创新高但量跟不上 → 顶背离，警惕")
        strength = 0.62
    elif ret5 < -2.0 and big_vol:
        state, direction = "下跌放量", DIR_BEAR
        sigs.append("下跌放量 → 抛压未尽，别抄")
        strength = 0.66
    elif ret5 < -0.5 and small_vol:
        state, direction = "缩量阴跌", DIR_BEAR
        sigs.append("缩量阴跌 → 没见底，只是流动性枯竭")
        strength = 0.5
    elif vol_dry and long_yang:
        state, direction = "地量后放量长阳", DIR_BULL
        sigs.append("地量后放量长阳 → 可能见底信号")
        strength = 0.7
    else:
        state = "量价中性"
        sigs.append("暂未出现明确的量价信号（量比 %.2f、位置 %.0f%%）" % (vr, pos * 100))

    out.update({"state": state, "direction": direction, "strength": strength,
                "reliability": 0.64, "signals": sigs,
                "vol_ratio": round(vr, 3), "vol_dry": bool(vol_dry),
                "position_pct": round(pos * 100, 1),
                "ret1_pct": round(ret1, 2), "ret5_pct": round(ret5, 2)})
    out["detail"] = ("量价验证：%s。量比（5日/20日）%.2f，股价位于一年区间 %.0f%% 位置。"
                     "口诀：低位放量不跌=建仓、涨放量调缩量=健康主升、"
                     "高位放量不涨=派发、价新高量跟不上=顶背离、"
                     "下跌放量=抛压未尽、缩量阴跌=没见底、地量后放量长阳=可能见底。"
                     % (out["signals"][0], vr, pos * 100))
    return out


# --------------------------------------------------------------------------- #
# 四、六波段识别
# --------------------------------------------------------------------------- #
def detect_phase(df: pd.DataFrame, regime: dict | None = None,
                 swing: dict | None = None, vp: dict | None = None) -> dict:
    """判定中级行情的六波段位置（牛初/牛中/牛末/熊初/熊中/熊末）。

    判据组合（每条都可观察，不是猜）：
      * 大周期方向（regime）
      * 位置：距高点/低点的距离、年内区间位置
      * 量能：是否天量/地量
      * 均线排列：多头/空头/缠绕
      * 量价验证结果
      * 波动率（天量天价常伴高波动）
    """
    out = {"phase": "", "label": "", "detail": "", "direction": DIR_NEUTRAL,
           "strength": 0.3, "reliability": 0.6, "traits": "", "signals": "",
           "play": ""}
    if df is None or not len(df):
        out["detail"] = "缺少数据，无法判定波段"
        return out
    regime = regime or detect_regime(df)
    swing = swing or swing_structure(df)
    vp = vp or volume_price_check(df)
    d = df.dropna(subset=["close"])
    c = d["close"].astype(float)
    v = d["volume"].astype(float) if "volume" in d.columns else pd.Series([1.0] * len(c))
    n = len(c)
    cur = float(c.iloc[-1])
    win = min(n, 250)
    hi, lo = float(c.tail(win).max()), float(c.tail(win).min())
    pos = (cur - lo) / max(hi - lo, 1e-9)
    ma20 = float(c.tail(20).mean())
    ma60 = float(c.tail(60).mean()) if n >= 60 else ma20
    ma250 = float(c.tail(250).mean()) if n >= 250 else ma60
    bull_ma = ma20 > ma60 > ma250
    bear_ma = ma20 < ma60 < ma250
    vr = float(v.tail(5).mean() / v.tail(20).mean()) if v.tail(20).mean() else 1.0
    shrink = float(v.tail(5).mean() / v.tail(120).mean()) if n >= 120 and v.tail(120).mean() else 1.0
    reg = regime.get("regime")
    st = swing.get("structure")
    state = vp.get("state")

    # ---- 打分式判定（避免单条规则误判）---- #
    score = {
        "bull_start": 0.0, "bull_mid": 0.0, "bull_end": 0.0,
        "bear_start": 0.0, "bear_mid": 0.0, "bear_end": 0.0,
    }
    # 牛初：低位 + 站上年线 + 温和放量 + 政策预期（用波动收敛代理）
    if pos < 0.45:
        score["bull_start"] += 2.0
    if cur > ma250:
        score["bull_start"] += 1.5
    if 0.9 <= vr <= 1.6:
        score["bull_start"] += 1.0
    if state == "低位放量不跌":
        score["bull_start"] += 2.0
    if reg == REGIME_BULL and pos < 0.5:
        score["bull_start"] += 1.5
    # 牛中：多头排列 + 位置中高 + 量价健康
    if bull_ma:
        score["bull_mid"] += 2.5
    if 0.45 <= pos <= 0.85:
        score["bull_mid"] += 1.5
    if state in ("上涨放量+回踩缩量", "低位放量不跌"):
        score["bull_mid"] += 1.5
    if reg == REGIME_BULL:
        score["bull_mid"] += 1.5
    if st == "上升波段":
        score["bull_mid"] += 1.0
    # 牛末：高位 + 天量 + 滞涨/背离
    if pos > 0.85:
        score["bull_end"] += 2.5
    if vr > 1.6:
        score["bull_end"] += 1.5
    if state in ("高位放量不涨", "价创新高但量跟不上"):
        score["bull_end"] += 2.5
    if reg == REGIME_BULL and pos > 0.8:
        score["bull_end"] += 1.5
    # 熊初：放量大跌 + 跌破均线 + 高位回落
    if state == "下跌放量":
        score["bear_start"] += 2.5
    if bear_ma:
        score["bear_start"] += 1.5
    if pos < 0.6:
        score["bear_start"] += 1.0
    if swing.get("structure") == "下降波段":
        score["bear_start"] += 1.0
    if regime.get("from_high_pct", 0) < -10:
        score["bear_start"] += 1.0
    # 熊中：空头排列 + 缩量阴跌 + 持续新低
    if bear_ma:
        score["bear_mid"] += 2.0
    if state == "缩量阴跌":
        score["bear_mid"] += 2.0
    if shrink < 0.85:
        score["bear_mid"] += 1.0
    if reg == REGIME_BEAR:
        score["bear_mid"] += 2.0
    if st == "下降波段":
        score["bear_mid"] += 1.0
    # 熊末：极低位 + 地量 + 长阳试探
    if pos < 0.2:
        score["bear_end"] += 2.5
    if vp.get("vol_dry"):
        score["bear_end"] += 1.5
    if state == "地量后放量长阳":
        score["bear_end"] += 2.5
    if reg == REGIME_BEAR and pos < 0.25:
        score["bear_end"] += 1.5

    # ---- 硬性前提闸门（"牛初/牛中"的定义要求站上关键均线）----
    # 曾经的问题：上证"跌破年线 7.4% + 三条均线全破"，却被判为"牛初（修复段）"。
    # 因为打分里"一年区间低位 +2.0"和"大周期牛市 +1.5"盖过了"未站上年线"。
    # 但用户给的框架里，**牛初的必备信号就是"站上年线、不破前低"**——
    # 不满足就不能叫牛初，否则标签本身就是骗人的。
    above_ma250 = cur >= ma250
    above_ma60 = cur >= ma60
    gates: list = []
    if not above_ma250:
        score["bull_start"] *= 0.25          # 未站上年线 → 牛初几乎不可能
        score["bull_mid"] *= 0.20            # 牛中更不可能（连中期均线都没站上）
        gates.append("未站上年线")
    if not above_ma60:
        score["bull_mid"] *= 0.30
        gates.append("未站上季线(MA60)")
    if bear_ma and not above_ma250:
        score["bear_start"] *= 1.25
        score["bear_mid"] *= 1.20            # 空头排列且破年线 → 熊市阶段更可能

    best = max(score.items(), key=lambda kv: kv[1])
    key, sc = best
    meta = PHASES[key]
    total = sum(score.values()) or 1.0
    conf = sc / total
    out.update({
        "phase": key, "label": meta["label"], "direction": meta["dir"],
        "strength": round(min(0.35 + conf * 1.1, 0.85), 3),
        "reliability": round(min(0.45 + conf * 0.9, 0.82), 3),
        "traits": meta["traits"], "signals": meta["signals"], "play": meta["play"],
        "scores": {k: round(v, 2) for k, v in score.items()},
        "confidence": round(conf, 3),
        "position_pct": round(pos * 100, 1),
        "ma_state": ("多头排列" if bull_ma else ("空头排列" if bear_ma else "均线缠绕")),
        # 与"大周期牛市"并列时的准确表述：避免"牛市 + 熊初"看似自相矛盾。
        # 大周期管方向（5 年），中期管节奏（月/季），级别不同故不冲突。
        # 置信度偏低时不给"确定"的口吻，避免把低置信当高置信用（AI 图4-1）
        "label_uncertain": bool(float(out.get("confidence") or 0) < 0.40),
        "label_ctx": (("牛市中的中级回调（%s）" % meta["label"])
                      if (("牛" in str(regime.get("regime") or ""))
                          and ("熊" in str(meta["label"])))
                      else meta["label"]),
        "gates": gates,
        "above_ma250": bool(above_ma250),
        "above_ma60": bool(above_ma60),
    })
    # 次优波段（波段常混合，给出参照）
    rivals = sorted(((k, v) for k, v in score.items() if k != key),
                    key=lambda kv: -kv[1])[:2]
    out["rivals"] = "、".join("%s(%.1f)" % (PHASES[k]["label"], v)
                             for k, v in rivals if v > 0)
    out["detail"] = (
        "中级波段判定为「%s」（置信度 %.0f%%）：%s。观察信号：%s。"
        "当前位置：一年区间 %.0f%%、%s、量比 %.2f。"
        % (meta["label"], conf * 100, meta["traits"], meta["signals"],
           pos * 100, out["ma_state"], vr))
    if gates:
        out["detail"] += (" 注意：当前%s，因此已压低「牛初/牛中」的可能性"
                          "——按定义，牛初必须站上年线且不破前低。"
                          % "、".join(gates))
    return out


# --------------------------------------------------------------------------- #
# 五、关键位置与操作纪律
# --------------------------------------------------------------------------- #
def key_levels(df: pd.DataFrame, swing: dict | None = None) -> dict:
    """给出关键观察位（支撑/压力）与操作纪律。"""
    out = {"detail": ""}
    if df is None or not len(df):
        return out
    d = df.dropna(subset=["close"])
    c = d["close"].astype(float)
    n = len(c)
    cur = float(c.iloc[-1])
    win = min(n, 250)
    hi, lo = float(c.tail(win).max()), float(c.tail(win).min())
    ma20 = float(c.tail(20).mean())
    ma60 = float(c.tail(60).mean()) if n >= 60 else ma20
    ma250 = float(c.tail(250).mean()) if n >= 250 else ma60
    supports = sorted([x for x in (ma20, ma60, ma250, lo) if x < cur], reverse=True)
    resists = sorted([x for x in (ma20, ma60, ma250, hi) if x > cur])
    out.update({
        "current": round(cur, 2),
        "support_1": round(supports[0], 2) if supports else None,
        "support_2": round(supports[1], 2) if len(supports) > 1 else None,
        "resistance_1": round(resists[0], 2) if resists else None,
        "resistance_2": round(resists[1], 2) if len(resists) > 1 else None,
        "ma20": round(ma20, 2), "ma60": round(ma60, 2), "ma250": round(ma250, 2),
        "year_high": round(hi, 2), "year_low": round(lo, 2),
        "range_position_pct": round((cur - lo) / max(hi - lo, 1e-9) * 100, 1),
    })
    out["detail"] = (
        "关键位置：当前 %.2f；下方支撑 %s / %s（MA20 %.2f、MA60 %.2f、MA250 %.2f）；"
        "上方压力 %s / %s（一年高点 %.2f）。"
        % (cur,
           out["support_1"] if out["support_1"] else "—",
           out["support_2"] if out["support_2"] else "—",
           ma20, ma60, ma250,
           out["resistance_1"] if out["resistance_1"] else "—",
           out["resistance_2"] if out["resistance_2"] else "—", hi))
    return out


#: 三层 × 三种短线结构 的操作纪律矩阵
DISCIPLINE = {
    ("上升波段",): (
        "**上升波段纪律**：缩量回踩关键均线可低吸，跌破就走。"
        "不要在放量长阳当天追高——等回踩缩量再买，成本低、风险小。"
    ),
    ("下降波段",): (
        "**下降波段纪律**：不接缩量反弹的飞刀。等「放量企稳」（放量长阳不破前低）"
        "再考虑，否则任何反弹都可能是下跌中继。"
    ),
    ("震荡波段",): (
        "**震荡波段纪律**：轻仓观望，等放量突破方向再顺势。"
        "震荡市里均线金叉死叉骗线极多，别单看指标——"
        "先看趋势结构，再看量，最后看指标。"
    ),
}


def operation_discipline(phase: dict, swing: dict, regime: dict,
                         vp: dict, levels: dict | None = None) -> dict:
    """综合三层给出**具体操作纪律**（不是一句看多看空）。"""
    st = swing.get("structure") or "震荡波段"
    ph = phase.get("phase") or ""
    out = {
        "position_advice": "",
        "entry": "",
        "exit": "",
        "avoid": "",
        "detail": "",
    }
    # ---- 仓位建议：由大周期与波段共同决定 ---- #
    base = 0.5
    if regime.get("regime") == REGIME_BULL:
        base = 0.7
    elif regime.get("regime") == REGIME_BEAR:
        base = 0.25
    delta = {"bull_start": 0.1, "bull_mid": 0.15, "bull_end": -0.3,
             "bear_start": -0.35, "bear_mid": -0.15, "bear_end": 0.05}.get(ph, 0.0)
    pos = max(0.0, min(1.0, base + delta))
    # ---- 均线健康度封顶 ----
    # 曾经的问题：大周期=牛市、波段=牛初 机械叠加就给出 80% 仓位，
    # 而实际「全部主要均线跌破 + 处于一年低位」时给出 80% 是危险的。
    # 规则：均线越差，仓位上限越低；波段置信度低时再打折。
    caps: list = []
    detail_cap = ""
    try:
        # 用 levels（已由 key_levels 算好），避免依赖 df 参数
        lv = levels or {}
        cur = float(lv.get("current") or 0)
        ma20 = float(lv.get("ma20") or 0)
        ma60 = float(lv.get("ma60") or 0)
        ma250 = float(lv.get("ma250") or 0)
        broken = sum(1 for m in (ma20, ma60, ma250) if m and cur < m)
        if broken >= 3:
            caps.append(0.35)
            detail_cap = "三条主要均线全部跌破"
        elif broken == 2:
            caps.append(0.55)
            detail_cap = "两条主要均线跌破"
        elif broken == 1:
            caps.append(0.75)
            detail_cap = "一条主要均线跌破"
        # 处于一年区间极低位时，不宜重仓抄底（等确认）
        rpos = float(lv.get("range_position_pct") or 50) / 100.0
        if rpos <= 0.10:
            caps.append(0.40)
            detail_cap += "；处于一年区间 %.0f%% 极低位" % (rpos * 100)
    except Exception:  # noqa: BLE001
        pass
    # 波段置信度低 → 整体打折
    try:
        cf = float(phase.get("confidence") or 0)
        if cf < 0.35:
            caps.append(pos * 0.7)
    except Exception:  # noqa: BLE001
        pass
    if caps:
        pos = min([pos] + caps)
    out["position_cap"] = detail_cap
    out["position_advice"] = "建议仓位约 %.0f%%（%s + %s%s）" % (
        pos * 100, regime.get("regime") or "—", phase.get("label") or "—",
        ("；已因" + detail_cap + "下调上限" if detail_cap else ""))
    # ---- 进出场 ---- #
    out["entry"] = (
        "只在**缩量回踩**关键支撑（MA20/MA60）时低吸；放量突破压力位可小仓跟进。"
        if st == "上升波段" else
        ("等**放量企稳**：地量后放量长阳、且回踩不破前低，才是右侧机会。"
         if st == "下降波段" else
         "区间下沿缩量买、上沿放量卖；不追中间位置。")
    )
    out["exit"] = (
        "跌破 MA20 减半、跌破 MA60 清仓；高位放量滞涨先走一半。"
        if st == "上升波段" else
        ("反弹到 MA20/MA60 附近且缩量，是减仓机会（反弹不是反转）。"
         if st == "下降波段" else
         "箱体上沿放量不过就减仓；跌破箱体下沿止损。")
    )
    out["avoid"] = (
        "不追高、不在放量长阳当天进场、不满仓单一板块。"
        if st == "上升波段" else
        ("不抄底、不摊平、不用杠杆；不因为「跌多了」就买。"
         if st == "下降波段" else
         "不在震荡中频繁交易（手续费与滑点会吃掉收益）；不重仓赌方向。")
    )
    disc = DISCIPLINE.get((st,), (DISCIPLINE[("震荡波段",)][0],))[0]
    vp_state = vp.get("state") or ""
    out["detail"] = (
        "%s\n\n**波段操作**：%s\n\n**仓位**：%s\n\n**进场**：%s\n\n**离场**：%s\n\n**禁忌**：%s"
        % (disc, phase.get("play") or "—", out["position_advice"],
           out["entry"], out["exit"], out["avoid"]))
    if vp_state:
        out["detail"] += "\n\n**当前量价信号**：%s" % vp_state
    return out


# --------------------------------------------------------------------------- #
# 六、因子
# --------------------------------------------------------------------------- #
def market_structure_factors(regime: dict, phase: dict, swing: dict,
                             vp: dict, levels: dict) -> list[Factor]:
    """把三层结构转成因子。"""
    out: list[Factor] = []
    if not regime.get("regime"):
        return out
    # 大周期
    out.append(Factor(
        id="market_regime_major", name="大周期：%s" % regime["regime"],
        category=Category.MACRO, direction=regime.get("direction", 0),
        strength=regime.get("strength", 0.3),
        reliability=regime.get("reliability", 0.6),
        value="距低点 %+.1f%%、距高点 %+.1f%%" % (
            regime.get("from_low_pct") or 0, regime.get("from_high_pct") or 0),
        detail=regime.get("detail") or "",
        theory="技术性牛熊定义（20% 阈值）/周期理论",
        horizon=("m1", "q1", "h1", "y1")))
    # 中级波段
    if phase.get("phase"):
        out.append(Factor(
            id="market_phase", name="中级波段：%s" % phase["label"],
            category=Category.MACRO, direction=phase.get("direction", 0),
            strength=phase.get("strength", 0.3),
            reliability=phase.get("reliability", 0.6),
            value="置信度 %.0f%%" % ((phase.get("confidence") or 0) * 100),
            detail="%s。观察信号：%s。操作：%s" % (
                phase.get("traits") or "", phase.get("signals") or "",
                phase.get("play") or ""),
            theory="六波段循环/道氏理论",
            horizon=("d20", "m1", "q1")))
    # 短线结构
    if swing.get("structure"):
        out.append(Factor(
            id="market_swing_structure", name="短线结构：%s" % swing["structure"],
            category=Category.TECH, direction=swing.get("direction", 0),
            strength=swing.get("strength", 0.2),
            reliability=swing.get("reliability", 0.55),
            value="高点%s、低点%s" % (swing.get("high_trend") or "—",
                                    swing.get("low_trend") or "—"),
            detail=swing.get("detail") or "",
            theory="趋势结构（高点低点序列）", horizon=("d1", "d5", "d10")))
    # 量价验证
    if vp.get("state"):
        out.append(Factor(
            id="market_volume_price", name="量价验证：%s" % vp["state"],
            category=Category.TECH, direction=vp.get("direction", 0),
            strength=vp.get("strength", 0.2), reliability=vp.get("reliability", 0.6),
            value="量比 %.2f、位置 %.0f%%" % (vp.get("vol_ratio") or 0,
                                            vp.get("position_pct") or 0),
            detail=vp.get("detail") or "",
            theory="量价关系（真假波段识别）", horizon=("d1", "d5", "d10", "d20")))
    # 关键位置
    if levels.get("support_1") or levels.get("resistance_1"):
        out.append(Factor(
            id="market_key_levels", name="关键位置",
            category=Category.MACRO, direction=DIR_NEUTRAL,
            strength=0.2, reliability=0.6,
            value="支撑 %s｜压力 %s" % (levels.get("support_1") or "—",
                                     levels.get("resistance_1") or "—"),
            detail=levels.get("detail") or "",
            theory="支撑压力/均线系统", horizon=("d5", "d10", "d20")))
    return out


# --------------------------------------------------------------------------- #
# 七、一站式入口
# --------------------------------------------------------------------------- #
def analyze_market_structure(index_df: pd.DataFrame, *,
                             name: str = "上证指数") -> dict:
    """三层嵌套完整分析：大周期 → 中级波段 → 短线结构 → 量价 → 操作纪律。"""
    out: dict = {"name": name, "regime": {}, "phase": {}, "swing": {},
                 "volume_price": {}, "levels": {}, "discipline": {},
                 "factors": [], "detail": "", "traits": A_SHARE_TRAITS,
                 "vp_rules": VP_RULES}
    if index_df is None or not len(index_df):
        out["detail"] = "缺少指数数据，无法做大盘结构分析"
        return out
    regime = detect_regime(index_df)
    swing = swing_structure(index_df)
    vp = volume_price_check(index_df)
    phase = detect_phase(index_df, regime, swing, vp)
    levels = key_levels(index_df, swing)
    disc = operation_discipline(phase, swing, regime, vp, levels)
    out.update({"regime": regime, "phase": phase, "swing": swing,
                "volume_price": vp, "levels": levels, "discipline": disc})
    # ---- 深度阶段判定（道氏+波浪+四维评分，不局限于六波段）----
    deep: dict = {}
    try:
        from . import market_phase as _MP

        deep = _MP.analyze_deep(index_df) or {}
        out["deep"] = {k: v for k, v in deep.items() if k != "factors"}
        out["deep_detail"] = deep.get("detail") or ""
    except Exception as exc:  # noqa: BLE001
        log.debug("深度阶段判定失败: %s", exc)
        out["deep"] = {}

    out["factors"] = market_structure_factors(regime, phase, swing, vp, levels)
    try:
        out["factors"] = list(out["factors"]) + (deep.get("factors") or [])
    except Exception:  # noqa: BLE001
        pass
    out["detail"] = (
        "三层嵌套判断 —— 大周期「%s」；中级波段「%s」；短线结构「%s」；"
        "量价「%s」。" % (regime.get("regime") or "—",
                        phase.get("label") or "—",
                        swing.get("structure") or "—",
                        vp.get("state") or "—"))
    return out


if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")

    from .datasource import fetch_daily
    from .selftest_data import make_ohlcv

    print("=== 1. 合成数据：牛市主升（持续上涨+放量）===")
    n = 500
    up = np.linspace(3000, 4200, n) + np.sin(np.arange(n) / 9) * 30
    df = make_ohlcv(n, seed=3, start_price=3000.0)
    df["close"] = up
    df["high"] = up * 1.012
    df["low"] = up * 0.988
    v = df["volume"].to_numpy(float).copy()
    v[:int(n * 0.7)] *= 0.8
    v[int(n * 0.7):] *= 1.6
    df["volume"] = v
    r = analyze_market_structure(df, name="合成-牛市")
    print("  ", r["detail"])
    print("   大周期:", r["regime"]["regime"], "距低点 %+.1f%%" % r["regime"]["from_low_pct"])
    print("   波段:", r["phase"]["label"], "置信度 %.0f%%" % (r["phase"]["confidence"] * 100))
    print("   短线:", r["swing"]["structure"], "| 量价:", r["volume_price"]["state"])
    print("   仓位:", r["discipline"]["position_advice"])

    print("\n=== 2. 合成数据：熊市阴跌（持续下跌+缩量）===")
    dn = np.linspace(4200, 3000, n) + np.sin(np.arange(n) / 8) * 25
    d2 = make_ohlcv(n, seed=7, start_price=4200.0)
    d2["close"] = dn
    d2["high"] = dn * 1.008
    d2["low"] = dn * 0.992
    v2 = d2["volume"].to_numpy(float).copy() * 0.55
    d2["volume"] = v2
    r2 = analyze_market_structure(d2, name="合成-熊市")
    print("  ", r2["detail"])
    print("   波段:", r2["phase"]["label"], "| 仓位:", r2["discipline"]["position_advice"])

    print("\n=== 3. 合成数据：震荡（区间来回）===")
    osc = 3500 + np.sin(np.arange(n) / 12) * 250
    d3 = make_ohlcv(n, seed=9, start_price=3500.0)
    d3["close"] = osc
    d3["high"] = osc * 1.01
    d3["low"] = osc * 0.99
    r3 = analyze_market_structure(d3, name="合成-震荡")
    print("   大周期:", r3["regime"]["regime"], "| 短线:", r3["swing"]["structure"])
    print("   波段:", r3["phase"]["label"])

    print("\n=== 4. 真实数据：上证指数 ===")
    try:
        real = fetch_daily("000001", days=1600)
        rr = analyze_market_structure(real, name="上证指数")
        print("  ", rr["detail"])
        print("   大周期:", rr["regime"]["detail"][:100])
        print("   波段:", rr["phase"]["label"], "置信度 %.0f%%" % (rr["phase"]["confidence"] * 100))
        print("   次优:", rr["phase"].get("rivals"))
        print("   短线:", rr["swing"]["detail"][:100])
        print("   量价:", rr["volume_price"]["state"])
        print("   位置:", rr["levels"]["detail"][:120])
        print("   因子 %d 条: %s" % (len(rr["factors"]), [f.name for f in rr["factors"]]))
        print("   操作纪律:")
        for line in rr["discipline"]["detail"].split("\n"):
            if line.strip():
                print("      ", line[:96])
    except Exception as exc:  # noqa: BLE001
        print("  真实数据获取失败（可跳过）:", str(exc)[:60])

    print("\n=== 5. 边界情况 ===")
    assert analyze_market_structure(None)["detail"]
    assert analyze_market_structure(pd.DataFrame())["detail"]
    for f in r["factors"]:
        assert 0 <= f.strength <= 1 and 0 <= f.reliability <= 1
        assert f.detail and f.theory and "None" not in str(f.value)
    print("  空输入安全；因子字段完整")
    print("\nmarket_structure self-test OK")
