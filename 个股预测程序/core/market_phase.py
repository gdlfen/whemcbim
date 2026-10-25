# -*- coding: utf-8 -*-
"""
深度阶段判定 —— 道氏理论 + 波浪理论 + A股策略语境 + 拉升回调分级。

为什么不满足于原有的六波段
--------------------------
原框架（牛初/牛中/牛末/熊初/熊中/熊末）是**粗粒度标签**，实际行情远比这复杂：
  · 同样是「上涨中」，1 浪启动、3 浪主升、5 浪末段的操作完全不同；
  · 同样是「回调」，回撤 20% 的洗盘和回撤 62% 的趋势破坏是两回事；
  · 同样是「熊市反弹」，反弹 15% 与反弹 55% 的性质截然不同。

所以本模块做**四维独立评分 + 精细分级**，而不是往六个格子里硬塞。

四维评分体系（每维 0~1，各自给证据与反证）
------------------------------------------
1. **核心驱动**：估值分位、两融趋势、北向资金、成交额水位
   —— A股是「政策+流动性晴雨表」，驱动决定方向，位置只是结果
2. **关键位置**：年线/季线、前高前低、斐波那契回撤位、缺口、整数关口
   —— 位置决定「这里是起点还是终点」
3. **量价关系**：量比、量价配合/背离、地量地价、天量天价
   —— 量价是波段的「真假识别器」
4. **结构形态**：道氏摆动点（HH/HL vs LH/LL）、波浪位置、均线排列
   —— 结构决定趋势是否成立

拉升与回调的「高度/深度」分级
-----------------------------
* 拉升高度：从波段起点涨幅 <15% 弱 / 15~30% 中 / 30~60% 强 / >60% 主升
* 回调深度（斐波那契）：<23.6% 强势洗盘 / 23.6~38.2% 正常 /
  38.2~50% 中期 / 50~61.8% 深度 / >61.8% 趋势可能反转
* 熊市反弹高度：<23.6% 弱反弹 / 23.6~38.2% 正常 / >50% 可能反转

道氏理论四要素（本模块全部量化）
--------------------------------
① 主要趋势由高低点序列定义（HH+HL = 上升，LH+LL = 下降）
② 次级折返必须区分「正常回撤」与「趋势破坏」（用斐波那契 + 均线）
③ 成交量必须确认趋势（上涨放量、下跌缩量=健康；背离=警惕）
④ 指数间相互印证（单指数分析时用板块宽度/多指数一致性代理）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import Category, DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Factor, safe_float
from .utils import get_logger

log = get_logger()

#: 斐波那契回撤档位
FIB_LEVELS = (0.236, 0.382, 0.500, 0.618, 0.786)

#: 拉升高度分级（从波段起点涨幅 %）
RALLY_GRADES = (
    (60.0, "主升级（>60%）", "通常是3浪或主升段，涨幅已大，追高风险上升"),
    (30.0, "强拉升（30~60%）", "趋势确立，但已进入中后段"),
    (15.0, "中拉升（15~30%）", "趋势形成初期，最具性价比"),
    (0.0, "弱拉升（<15%）", "尚属修复或试探，未成趋势"),
)

#: 回调深度分级（回撤比例 0~1）
PULLBACK_GRADES = (
    (0.786, "超深回调（>78.6%）", "趋势大概率已被破坏，需按反转对待"),
    (0.618, "深度回调（61.8~78.6%）", "已破黄金分割半分位，趋势存疑"),
    (0.500, "偏深回调（50~61.8%）", "多空分水岭，此处失守则趋势转弱"),
    (0.382, "中期回调（38.2~50%）", "正常调整范围的上限"),
    (0.236, "正常回调（23.6~38.2%）", "健康调整，趋势未受实质影响"),
    (0.0, "强势洗盘（<23.6%）", "极强，回调极浅，多头掌控"),
)


def _grade(value: float, table) -> tuple[str, str]:
    """按阈值表给分级（表按阈值降序）。"""
    for thr, name, desc in table:
        if value >= thr:
            return name, desc
    return table[-1][1], table[-1][2]


# --------------------------------------------------------------------------- #
# 一、摆动点与波浪结构
# --------------------------------------------------------------------------- #
def swing_points(df: pd.DataFrame, order: int = 5, lookback: int = 400) -> dict:
    """提取 ZigZag 摆动点，并判定道氏结构（HH/HL vs LH/LL）。"""
    out = {"points": [], "structure": "不明", "n_highs": 0, "n_lows": 0,
           "hh": 0, "hl": 0, "lh": 0, "ll": 0, "detail": ""}
    if df is None or not len(df) or "high" not in df.columns:
        out["detail"] = "缺少数据"
        return out
    d = df.tail(lookback).reset_index(drop=True)
    hi = d["high"].astype(float).to_numpy()
    lo = d["low"].astype(float).to_numpy()
    n = len(hi)
    if n < order * 6:
        out["detail"] = "样本不足，无法提取摆动点"
        return out
    pts: list = []
    for i in range(order, n - order):
        if hi[i] == max(hi[i - order:i + order + 1]):
            pts.append((i, float(hi[i]), "H"))
        elif lo[i] == min(lo[i - order:i + order + 1]):
            pts.append((i, float(lo[i]), "L"))
    out["points"] = [{"idx": p[0], "price": round(p[1], 2), "type": p[2]}
                     for p in pts[-12:]]
    highs = [p[1] for p in pts if p[2] == "H"]
    lows = [p[1] for p in pts if p[2] == "L"]
    out["n_highs"], out["n_lows"] = len(highs), len(lows)
    if len(highs) >= 2 and len(lows) >= 2:
        hh = sum(1 for a, b in zip(highs, highs[1:]) if b > a)
        hl = sum(1 for a, b in zip(lows, lows[1:]) if b > a)
        lh = sum(1 for a, b in zip(highs, highs[1:]) if b < a)
        ll = sum(1 for a, b in zip(lows, lows[1:]) if b < a)
        out.update({"hh": hh, "hl": hl, "lh": lh, "ll": ll})
        if hh >= 1 and hl >= 1 and hh + hl > lh + ll:
            out["structure"] = "上升结构（高点抬高+低点抬高）"
        elif lh >= 1 and ll >= 1 and lh + ll > hh + hl:
            out["structure"] = "下降结构（高点降低+低点降低）"
        else:
            out["structure"] = "收敛/震荡（高低点方向不一致）"
        out["detail"] = ("摆动点 %d 个（%d高 %d低）：高点上移 %d 次/下移 %d 次，"
                         "低点上移 %d 次/下移 %d 次 → %s"
                         % (len(pts), len(highs), len(lows), hh, lh, hl, ll,
                            out["structure"]))
    else:
        out["detail"] = "摆动点不足 2 高 2 低，结构不明"
    return out


def _last_swing(df: pd.DataFrame, sp: dict) -> dict:
    """找最近一段完整摆动（起点→极值→当前），用于算拉升高度与回调深度。"""
    out = {"rally_pct": None, "pullback_ratio": None, "start_idx": None,
           "peak_idx": None, "direction": ""}
    if df is None or not len(df) or not sp.get("points"):
        return out
    d = df.reset_index(drop=True)
    closes = d["close"].astype(float).to_numpy()
    n = len(closes)
    pts = sp["points"]
    base = n - len(d) if False else 0          # tail 起点（点了 tail 后 idx 是相对的）
    # sp 的 idx 是 tail(lookback) 之后的相对索引；换算到 d 的绝对索引
    off = max(0, len(d) - 400)
    abs_pts = [(int(p["idx"]) + off, float(p["price"]), p["type"]) for p in pts]
    last_idx, last_px, last_type = abs_pts[-1]
    # 最近一个与末点类型相反的点作为摆动起点
    start = None
    for p in reversed(abs_pts[:-1]):
        if p[2] != last_type:
            start = p
            break
    if start is None:
        return out
    cur = float(closes[-1])
    # 摆动幅度下限：不足峰值 1% 的摆动视为"无有效摆动"，否则回撤比值会被
    # 分母过小放大到 100% 以上（曾出现 222% 这种不可能的数）。
    MIN_SWING_PCT = 0.01
    if start[2] == "L" and last_type == "H":
        # 上升摆动：低点 → 高点，然后可能回调
        rng = last_px - start[1]
        out["start_idx"], out["peak_idx"] = start[0], last_idx
        out["direction"] = "上升摆动"
        if start[1] > 0:
            out["rally_pct"] = round((last_px / start[1] - 1) * 100, 2)
        if rng > 0 and last_px > 0 and (rng / last_px) >= MIN_SWING_PCT:
            r = (last_px - cur) / rng
            if r > 1.0:
                out["pullback_note"] = "已跌破摆动起点（回撤>100%，趋势可能已反转）"
            out["pullback_ratio"] = round(min(max(r, 0.0), 1.0), 4)
        elif rng > 0:
            out["pullback_note"] = "摆动幅度不足 %.0f%%，回撤比例不计算" % (
                MIN_SWING_PCT * 100)
    elif start[2] == "H" and last_type == "L":
        # 下降摆动：高点 → 低点，然后可能反弹
        rng = start[1] - last_px
        out["start_idx"], out["peak_idx"] = start[0], last_idx
        out["direction"] = "下降摆动"
        if start[1] > 0:
            out["rally_pct"] = round((cur / last_px - 1) * 100, 2)   # 反弹高度
        if rng > 0 and start[1] > 0 and (rng / start[1]) >= MIN_SWING_PCT:
            r = (cur - last_px) / rng
            if r > 1.0:
                out["pullback_note"] = "已收复下跌起点（反弹>100%，可能已是反转）"
            out["pullback_ratio"] = round(min(max(r, 0.0), 1.0), 4)
        elif rng > 0:
            out["pullback_note"] = "摆动幅度不足 %.0f%%，反弹比例不计算" % (
                MIN_SWING_PCT * 100)
    return out


# --------------------------------------------------------------------------- #
# 二、四维评分
# --------------------------------------------------------------------------- #
def score_drivers(df: pd.DataFrame, fund: dict | None = None) -> dict:
    """① 核心驱动分：估值分位 + 两融趋势 + 北向 + 成交额水位。"""
    out = {"score": 0.5, "items": [], "detail": ""}
    parts: list = []
    f = fund or {}
    pe = f.get("pe") or {}
    pct = pe.get("percentile")
    if pct is not None:
        # 估值分位越低越偏多（-1~1）
        v = (50 - float(pct)) / 50.0
        parts.append(("估值分位 %.0f%%" % pct, v, 0.9))
    mg = f.get("margin") or {}
    m20 = mg.get("change_20d_pct")
    if m20 is not None:
        v = max(-1.0, min(1.0, float(m20) / 10.0))
        parts.append(("两融近20日 %+.1f%%" % m20, v, 0.85))
    nt = f.get("north") or {}
    s20 = nt.get("sum_20d")
    if s20 is not None:
        v = max(-1.0, min(1.0, float(s20) / 400.0))
        parts.append(("北向近20日 %+.0f亿" % s20, v, 0.7))
    # 成交额水位（放量=活跃，缩量=清淡），用成交额 20 日均值与 120 日均值比
    try:
        amt = df["amount"].astype(float) if "amount" in df.columns else None
        if amt is None or not amt.notna().any():
            amt = (df["close"].astype(float) * df["volume"].astype(float)) \
                if "volume" in df.columns else None
        if amt is not None:
            a = amt.dropna()
            if len(a) >= 120:
                r = float(a.tail(20).mean() / a.tail(120).mean())
                v = max(-1.0, min(1.0, (r - 1.0) * 2.0))
                parts.append(("成交额20日/120日 %.2f" % r, v, 0.6))
    except Exception:  # noqa: BLE001
        pass
    if not parts:
        out["detail"] = "核心驱动数据不足（估值/两融/北向均缺失）"
        return out
    tot_w = sum(w for _, _, w in parts)
    s = sum(v * w for _, v, w in parts) / max(tot_w, 1e-9)
    out["score"] = round((s + 1) / 2, 3)          # 映射到 0~1
    out["items"] = [(n, round(v, 2)) for n, v, _ in parts]
    out["detail"] = "核心驱动：%s → 偏%s（%.0f/100）" % (
        "；".join("%s(%+.2f)" % (n, v) for n, v in out["items"]),
        "多" if s > 0.15 else ("空" if s < -0.15 else "中性"),
        out["score"] * 100)
    return out


def score_position(df: pd.DataFrame, levels: dict | None = None) -> dict:
    """② 关键位置分：年线/季线、一年区间、斐波那契位、缺口。"""
    out = {"score": 0.5, "items": [], "detail": ""}
    if df is None or not len(df):
        out["detail"] = "缺少数据"
        return out
    c = df["close"].astype(float).dropna()
    if len(c) < 60:
        out["detail"] = "样本不足"
        return out
    cur = float(c.iloc[-1])
    parts: list = []
    for bars, name, w in ((250, "年线", 1.0), (120, "半年线", 0.7), (60, "季线", 0.6),
                          (20, "月线", 0.4)):
        if len(c) >= bars:
            ma = float(c.tail(bars).mean())
            gap = (cur / ma - 1) * 100
            v = max(-1.0, min(1.0, gap / 5.0))     # 偏离 ±5% 达饱和
            parts.append(("%s %+.1f%%" % (name, gap), v, w))
    win = min(len(c), 250)
    hi, lo = float(c.tail(win).max()), float(c.tail(win).min())
    pos = (cur - lo) / max(hi - lo, 1e-9)
    # 位置分：中低位偏多（有空间）、极高位偏空（透支）。
    # 但**趋势感知**：在明确上升趋势中，高位是确认而非透支，惩罚减半。
    above_year = cur >= (float(c.tail(250).mean()) if len(c) >= 250
                         else float(c.tail(60).mean()))
    if pos > 0.5:
        vpos = 1.0 - 2.0 * pos
        if above_year:
            vpos *= 0.5            # 年线上方的高位：不再重罚
    else:
        vpos = 0.6 - pos
    # 权重从 0.8 降到 0.35 —— 位置只作辅助，主要趋势由年线与均线定义
    parts.append(("一年区间 %.0f%%" % (pos * 100), max(-1.0, min(1.0, vpos)), 0.35))
    tot = sum(w for _, _, w in parts)
    s = sum(v * w for _, v, w in parts) / max(tot, 1e-9)
    out["score"] = round((s + 1) / 2, 3)
    out["items"] = [(n, round(v, 2)) for n, v, _ in parts]
    out["detail"] = "关键位置：%s → 偏%s（%.0f/100）" % (
        "；".join("%s(%+.2f)" % (n, v) for n, v in out["items"]),
        "多" if s > 0.15 else ("空" if s < -0.15 else "中性"), out["score"] * 100)
    return out


def score_volume_price(df: pd.DataFrame) -> dict:
    """③ 量价分：量比、量价配合/背离、天量地量。"""
    out = {"score": 0.5, "items": [], "detail": "", "state": ""}
    if df is None or not len(df) or "volume" not in df.columns:
        out["detail"] = "缺少量能数据"
        return out
    d = df.dropna(subset=["close", "volume"])
    if len(d) < 60:
        out["detail"] = "样本不足"
        return out
    c = d["close"].astype(float).to_numpy()
    v = d["volume"].astype(float).to_numpy()
    n = len(c)
    vr = float(np.mean(v[-5:]) / max(np.mean(v[-20:]), 1e-9))
    ret5 = (c[-1] / c[-6] - 1) * 100 if n > 6 else 0.0
    win = min(n, 250)
    hi, lo = float(np.max(c[-win:])), float(np.min(c[-win:]))
    pos = (c[-1] - lo) / max(hi - lo, 1e-9)
    parts: list = []
    state = "量价中性"
    if pos < 0.35 and vr > 1.35 and ret5 > -1.0:
        parts.append(("低位放量不跌", 0.9, 1.0)); state = "低位放量不跌（建仓）"
    elif ret5 > 1.5 and vr > 1.35:
        parts.append(("上涨放量", 0.85, 1.0)); state = "上涨放量（健康主升）"
    elif pos > 0.70 and vr > 1.35 and ret5 < 1.0:
        parts.append(("高位放量滞涨", -0.9, 1.0)); state = "高位放量不涨（派发）"
    elif ret5 < -2.0 and vr > 1.35:
        parts.append(("下跌放量", -0.85, 1.0)); state = "下跌放量（抛压未尽）"
    elif ret5 < -0.5 and vr < 0.75:
        parts.append(("缩量阴跌", -0.6, 0.9)); state = "缩量阴跌（未止跌）"
    # 量价相关性（趋势的量能确认）
    try:
        if n >= 60:
            sub = d.tail(60)
            cc = sub["close"].astype(float)
            vv = sub["volume"].astype(float)
            corr = float(cc.corr(vv))
            if corr == corr:
                parts.append(("量价相关 %.2f" % corr, max(-1.0, min(1.0, corr * 1.5)), 0.5))
    except Exception:  # noqa: BLE001
        pass
    if not parts:
        parts.append(("量比 %.2f" % vr, max(-1.0, min(1.0, (vr - 1.0))), 0.5))
    tot = sum(w for _, _, w in parts)
    s = sum(x * w for _, x, w in parts) / max(tot, 1e-9)
    out["score"] = round((s + 1) / 2, 3)
    out["items"] = [(n2, round(x, 2)) for n2, x, _ in parts]
    out["state"] = state
    out["detail"] = "量价：%s；%s → 偏%s（%.0f/100）" % (
        state, "；".join("%s(%+.2f)" % (n2, x) for n2, x in out["items"]),
        "多" if s > 0.15 else ("空" if s < -0.15 else "中性"), out["score"] * 100)
    return out


def score_structure(df: pd.DataFrame, sp: dict | None = None) -> dict:
    """④ 结构分：道氏摆动点 + 均线排列 + 波浪位置（回撤深度）。"""
    out = {"score": 0.5, "items": [], "detail": "", "wave_hint": ""}
    if df is None or not len(df):
        out["detail"] = "缺少数据"
        return out
    sp = sp or swing_points(df)
    c = df["close"].astype(float).dropna()
    if len(c) < 60:
        out["detail"] = "样本不足"
        return out
    cur = float(c.iloc[-1])
    ma20 = float(c.tail(20).mean())
    ma60 = float(c.tail(60).mean())
    ma250 = float(c.tail(250).mean()) if len(c) >= 250 else ma60
    parts: list = []
    st = sp.get("structure") or ""
    if "上升结构" in st:
        parts.append(("道氏:HH+HL", 0.85, 1.0))
    elif "下降结构" in st:
        parts.append(("道氏:LH+LL", -0.85, 1.0))
    else:
        parts.append(("道氏:收敛", 0.0, 0.7))
    if ma20 > ma60 > ma250:
        parts.append(("均线多头排列", 0.8, 0.9))
    elif ma20 < ma60 < ma250:
        parts.append(("均线空头排列", -0.8, 0.9))
    else:
        parts.append(("均线缠绕", 0.0, 0.6))
    # 波浪位置提示（用回撤深度推断）
    sw = _last_swing(df, sp)
    pr = sw.get("pullback_ratio")
    hint = ""
    if pr is not None and sw.get("direction") == "上升摆动":
        if pr >= 0.618:
            hint = "回撤超 61.8%，可能进入调整浪 C 或趋势反转"
            parts.append(("回撤 %.0f%%（超深）" % (pr * 100), -0.8, 0.8))
        elif pr <= 0.236:
            hint = "回撤极浅，可能是推动浪中的次级调整（2/4 浪）"
            parts.append(("回撤 %.0f%%（强势）" % (pr * 100), 0.55, 0.7))
        else:
            hint = "回撤 %.0f%%，属正常调整浪" % (pr * 100)
            parts.append(("回撤 %.0f%%" % (pr * 100), 0.15, 0.6))
    elif pr is not None and sw.get("direction") == "下降摆动":
        if pr >= 0.5:
            hint = "反弹已收复 50% 以上，可能不是反弹而是反转（B 浪扩张/新趋势）"
            parts.append(("反弹收复 %.0f%%" % (pr * 100), 0.6, 0.8))
        else:
            hint = "反弹仅收复 %.0f%%，符合下跌中继的 B 浪反弹" % (pr * 100)
            parts.append(("反弹收复 %.0f%%" % (pr * 100), -0.2, 0.7))
    out["wave_hint"] = hint
    tot = sum(w for _, _, w in parts)
    s = sum(x * w for _, x, w in parts) / max(tot, 1e-9)
    score = (s + 1) / 2
    # ---- 道氏主要趋势闸门 ----
    # 道氏理论：主要趋势由**长期结构**定义，短期摆动不能改变主要趋势方向。
    # 价格在年线下方时，即使近期造出 HH+HL（多为反弹），结构分不得给偏高。
    below_year = cur < ma250
    bear_ma = ma20 < ma60 < ma250
    bull_ma = ma20 > ma60 > ma250
    gate_note = ""
    if below_year:
        score = min(score, 0.45)
        gate_note = "价格在年线下方，结构分上限 0.45（道氏：主要趋势仍偏下）"
    if below_year and bear_ma:
        score = min(score, 0.38)
        gate_note = "年线下方且均线空头排列，结构分上限 0.38"
    if (not below_year) and bull_ma:
        score = max(score, 0.62)
        gate_note = "年线上方且均线多头排列，结构分下限 0.62"
    out["score"] = round(score, 3)
    if gate_note:
        out["detail"] = gate_note + "。" + out.get("detail", "")
    out["items"] = [(n2, round(x, 2)) for n2, x, _ in parts]
    out["detail"] = "结构：%s → 偏%s（%.0f/100）" % (
        "；".join("%s(%+.2f)" % (n2, x) for n2, x in out["items"]),
        "多" if s > 0.15 else ("空" if s < -0.15 else "中性"), out["score"] * 100)
    return out


# --------------------------------------------------------------------------- #
# 三、合成与精细阶段
# --------------------------------------------------------------------------- #
#: 四维权重：驱动最重（定方向），结构次之（判成立），位置与量价辅助
DIM_WEIGHTS = {"driver": 0.34, "structure": 0.28, "position": 0.20, "volume": 0.18}


def compose(driver: dict, position: dict, volume: dict, structure: dict,
            trend_ctx: dict | None = None) -> dict:
    """四维合成 → 牛熊倾向 + 置信度 + 一致性。

    ``trend_ctx`` 提供道氏主要趋势上下文（``below_year`` / ``bear_ma`` /
    ``bull_ma``），用于施加全局闸门——避免"短期反弹"把总分推到"偏多"。
    """
    dims = {"driver": driver.get("score", 0.5), "structure": structure.get("score", 0.5),
            "position": position.get("score", 0.5), "volume": volume.get("score", 0.5)}
    total = sum(dims[k] * DIM_WEIGHTS[k] for k in dims)
    gate = ""
    tc = trend_ctx or {}
    if tc.get("below_year") and tc.get("bear_ma"):
        if total > 0.44:
            gate = "年线下方且均线空头排列 → 总分上限 0.44（不可判偏多）"
        total = min(total, 0.44)
    elif tc.get("below_year"):
        if total > 0.50:
            gate = "价格在年线下方 → 总分上限 0.50（主要趋势未修复）"
        total = min(total, 0.50)
    elif tc.get("bull_ma"):
        if total < 0.56:
            gate = "年线上方且均线多头排列 → 总分下限 0.56（主要趋势向上）"
        total = max(total, 0.56)
    vals = list(dims.values())
    spread = max(vals) - min(vals)
    # 一致性：四维越接近，置信度越高；分歧大则降低
    agree = 1.0 - min(spread / 0.6, 1.0)
    lean = "偏多" if total >= 0.56 else ("偏空" if total <= 0.44 else "中性")
    conf = round(0.35 + agree * 0.45, 3)
    return {"total": round(total, 3), "lean": lean, "confidence": conf,
            "agreement": round(agree, 3), "spread": round(spread, 3),
            "gate": gate, "dims": {k: round(v, 3) for k, v in dims.items()}}


def fine_phase(compose_r: dict, driver: dict, position: dict, volume: dict,
               structure: dict, swing: dict, df: pd.DataFrame) -> dict:
    """精细阶段判定 —— 不局限于六波段。"""
    total = compose_r["total"]
    lean = compose_r["lean"]
    dims = compose_r["dims"]
    sp_st = swing.get("structure") or ""
    up_struct = "上升结构" in sp_st
    dn_struct = "下降结构" in sp_st
    sw = _last_swing(df, swing)
    pr = sw.get("pullback_ratio")
    rally = sw.get("rally_pct")
    vstate = volume.get("state") or ""

    c = df["close"].astype(float).dropna()
    cur = float(c.iloc[-1])
    ma250 = float(c.tail(250).mean()) if len(c) >= 250 else float(c.tail(60).mean())
    above_year = cur >= ma250
    win = min(len(c), 250)
    hi, lo = float(c.tail(win).max()), float(c.tail(win).min())
    pos = (cur - lo) / max(hi - lo, 1e-9)

    stage = "方向未定的震荡整理"
    advice = ""
    wave = ""
    level = ""

    # 道氏原则：**主要趋势由结构定义，优先于综合分**。
    # 综合分刚好落在 0.5 附近但结构明确向上/向下时，应按结构判定方向，
    # 否则强势上涨会被误标成"顶部区域构筑"。
    # 摆动点有时不足（单边行情摆动点太少），此时用均线排列作为结构的替代判据
    _ma20 = float(c.tail(20).mean())
    _ma60 = float(c.tail(60).mean())
    _bull_ma = _ma20 > _ma60 > ma250
    _bear_ma = _ma20 < _ma60 < ma250
    # 季线方向：20 根之前的 MA60 与当前 MA60 比较。
    # 浅回调会让 MA20 短暂下穿 MA60（破坏多头排列），但季线仍在上升，
    # 所以用"季线方向"作为趋势的更稳定判据。
    _ma60_prev = float(c.tail(80).head(20).mean()) if len(c) >= 80 else _ma60
    _ma60_rising = _ma60 > _ma60_prev
    _ma60_falling = _ma60 < _ma60_prev
    uptrend = above_year and (up_struct or _bull_ma or _ma60_rising)
    downtrend = (not above_year) and (dn_struct or _bear_ma or _ma60_falling)
    if uptrend:
        if up_struct and (rally is None or rally < 30):
            stage = "上升趋势早段（推动浪 1 或 3 的初段）"
            wave = "推动浪（1/3 浪）初段"
            advice = "趋势刚形成，回踩不破关键均线可加仓；这是性价比最高的阶段"
        elif up_struct and rally is not None and rally >= 60:
            stage = "上升趋势后段（3 浪末或 5 浪）"
            wave = "推动浪末段（3末/5浪）"
            advice = "涨幅已大，改为持股+移动止盈，不再新开大仓位"
        else:
            stage = "上升趋势中继（调整浪末端或 4 浪）"
            wave = "调整浪（2/4 浪）"
            advice = "等回调结束信号（缩量止跌+放量收复）再介入"
    elif downtrend:
        if dn_struct and (pr is None or pr < 0.382):
            stage = "下跌趋势延续（下跌推动浪，A 或 C 浪）"
            wave = "下跌推动（A/C 浪）"
            advice = "不抄底；等放量止跌或地量长阳再评估"
        elif dn_struct and pr is not None and pr >= 0.5:
            stage = "下跌中的强反弹（B 浪扩张，需防反转）"
            wave = "B 浪反弹（可能转为新趋势）"
            advice = "反弹已收复一半以上跌幅，性质可能已变，关注能否站上年线"
        else:
            stage = "下跌趋势中的弱反弹（B 浪）"
            wave = "B 浪弱反弹"
            advice = "反弹无量、高点降低，属下跌中继，反弹是减仓机会"
    else:
        if pos <= 0.15:
            stage = "底部区域构筑（多重底/圆弧底可能）"
            wave = "底部形态（未确认）"
            advice = "等「地量→放量长阳→回踩不破前低」三步确认，再右侧介入"
        elif pos >= 0.85:
            stage = "顶部区域构筑（放量滞涨/顶背离风险）"
            wave = "顶部形态（未确认）"
            advice = "不追高；若出现高位放量不涨，先减仓"
        else:
            stage = "区间震荡（箱体整理，方向待选）"
            wave = "整理形态"
            advice = "轻仓等突破方向；不要在中位追涨杀跌"

    return {"stage": stage, "wave": wave, "advice": advice,
            "above_year": bool(above_year), "range_pos": round(pos * 100, 1),
            "rally_pct": rally, "pullback_ratio": pr,
            "detail": "%s；%s；%s" % (stage, wave or "浪型未明", advice)}


# --------------------------------------------------------------------------- #
# 四、因子
# --------------------------------------------------------------------------- #
def phase_factors(compose_r: dict, fine: dict, dims: dict) -> list:
    out: list[Factor] = []
    total = compose_r["total"]
    d = 1 if total >= 0.56 else (-1 if total <= 0.44 else 0)
    out.append(Factor(
        "deep_phase", "精细阶段：%s" % fine["stage"], Category.MACRO, d,
        min(abs(total - 0.5) * 3.2 + 0.25, 0.85), compose_r["confidence"],
        value="四维合成 %.0f/100（%s）" % (total * 100, compose_r["lean"]),
        detail="四维评分：核心驱动 %.0f、结构 %.0f、位置 %.0f、量价 %.0f（0~100）。"
               "阶段判定：%s。浪型：%s。策略：%s"
               % (dims["driver"] * 100, dims["structure"] * 100,
                  dims["position"] * 100, dims["volume"] * 100,
                  fine["stage"], fine["wave"] or "未明", fine["advice"]),
        theory="道氏理论+波浪理论+量价结构", horizon=("d20", "m1", "q1")))
    # 拉升高度 / 回调深度分级
    r = fine.get("rally_pct")
    pr = fine.get("pullback_ratio")
    if r is not None:
        name, desc = _grade(abs(r), RALLY_GRADES)
        out.append(Factor(
            "deep_rally_grade", "拉升高度：%s" % name, Category.TECH,
            DIR_BULL if r > 0 else DIR_BEAR, 0.4, 0.62,
            value="%+.2f%%" % r, detail="%s。%s" % (name, desc),
            theory="波段涨幅分级", horizon=("d5", "d10", "d20")))
    if pr is not None:
        # 回撤夹取到 1.0 时语义已不是"超深回调"而是"完全回吐"——
        # AI 核对指出"回撤100%却标超深回调"自相矛盾，这里单独表述。
        if pr >= 0.999:
            name, desc = ("完全回吐（回撤 100%）",
                          "涨幅已全部吐回甚至跌破起点，不属常规级别回调，"
                          "应按趋势可能反转对待，不宜继续套用斐波那契档位")
        else:
            name, desc = _grade(pr, PULLBACK_GRADES)
        out.append(Factor(
            "deep_pullback_grade", "回调深度：%s" % name, Category.TECH,
            DIR_BULL if pr <= 0.382 else (DIR_BEAR if pr >= 0.5 else DIR_NEUTRAL),
            0.5, 0.64, value="回撤 %.1f%%" % (pr * 100),
            detail="%s。%s（斐波那契档位：23.6/38.2/50/61.8）" % (name, desc),
            theory="斐波那契回撤/波浪理论", horizon=("d5", "d10", "d20")))
    return out


# --------------------------------------------------------------------------- #
# 五、一站式
# --------------------------------------------------------------------------- #
def analyze_deep(df: pd.DataFrame, fund: dict | None = None) -> dict:
    """深度阶段判定：四维评分 + 精细阶段 + 拉升回调分级。"""
    out: dict = {"swing": {}, "driver": {}, "position": {}, "volume": {},
                 "structure": {}, "compose": {}, "fine": {}, "factors": [],
                 "detail": ""}
    if df is None or not len(df) or len(df) < 60:
        out["detail"] = "数据不足 60 根，无法做深度阶段判定"
        return out
    sp = swing_points(df)
    driver = score_drivers(df, fund)
    position = score_position(df)
    volume = score_volume_price(df)
    structure = score_structure(df, sp)
    # 道氏主要趋势上下文（供合成阶段施加全局闸门）
    _c = df["close"].astype(float).dropna()
    _cur = float(_c.iloc[-1])
    _ma20 = float(_c.tail(20).mean())
    _ma60 = float(_c.tail(60).mean())
    _ma250 = float(_c.tail(250).mean()) if len(_c) >= 250 else _ma60
    trend_ctx = {"below_year": _cur < _ma250,
                 "bear_ma": _ma20 < _ma60 < _ma250,
                 "bull_ma": _ma20 > _ma60 > _ma250}
    comp = compose(driver, position, volume, structure, trend_ctx)
    fine = fine_phase(comp, driver, position, volume, structure, sp, df)
    out.update({"swing": sp, "driver": driver, "position": position,
                "volume": volume, "structure": structure, "compose": comp,
                "fine": fine})
    out["factors"] = phase_factors(comp, fine, comp["dims"])
    if comp.get("gate"):
        comp["detail_gate"] = comp["gate"]
    out["detail"] = ("四维合成 %.0f/100（%s，置信度 %.0f%%，四维一致度 %.0f%%）｜"
                     "阶段：%s｜%s%s"
                     % (comp["total"] * 100, comp["lean"], comp["confidence"] * 100,
                        comp["agreement"] * 100, fine["stage"],
                        fine["wave"] or "浪型未明",
                        ("｜闸门：" + comp["gate"]) if comp.get("gate") else ""))
    return out


if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")

    from .selftest_data import make_ohlcv

    def mk(path, seed=3, vol_mult=1.0):
        n = len(path)
        df = make_ohlcv(n, seed=seed, start_price=float(path[0]))
        c = np.asarray(path, dtype=float)
        df["close"] = c
        df["high"] = c * 1.008
        df["low"] = c * 0.992
        v = df["volume"].to_numpy(float).copy() * vol_mult
        df["volume"] = v
        df["amount"] = v * c
        return df

    print("=== 1. 强势上涨后浅回调（应判上升趋势，回调属洗盘级）===")
    n1 = 400
    # 涨 60% 后回调 6%（极浅，收在均线上方）—— 无歧义的强势上升趋势
    p1 = list(np.linspace(100, 100, 120)) + list(np.linspace(100, 160, 260)) \
        + list(np.linspace(160, 156, 20))       # 只回调 20 根（季线仍上行）
    d1 = mk(p1, seed=3)
    r1 = analyze_deep(d1)
    print(" ", r1["detail"])
    print("  回调深度:", [f.name for f in r1["factors"] if "回调" in f.name])
    print("  拉升高度:", [f.name for f in r1["factors"] if "拉升" in f.name])

    print("\n=== 2. 下跌趋势中的深跌后强反弹（应判B浪，防反转）===")
    # 跌 37.5% 后仅反弹 12%（弱反弹，仍在年线下方）—— 无歧义的下跌趋势
    p2 = list(np.linspace(160, 160, 120)) + list(np.linspace(160, 100, 180)) \
        + list(np.linspace(100, 112, 100))
    d2 = mk(p2, seed=7)
    r2 = analyze_deep(d2)
    print(" ", r2["detail"])
    print("  阶段:", r2["fine"]["stage"])
    print("  浪型:", r2["fine"]["wave"])

    print("\n=== 3. 低位窄幅震荡（应判底部构筑/区间震荡）===")
    p3 = list(100 + np.sin(np.arange(400) / 30) * 3)
    d3 = mk(p3, seed=9)
    r3 = analyze_deep(d3)
    print(" ", r3["detail"])
    print("  阶段:", r3["fine"]["stage"])
    print("  一年区间位置: %.1f%%" % r3["fine"]["range_pos"])

    print("\n=== 4. 校验 ===")
    for r in (r1, r2, r3):
        c = r["compose"]
        assert 0.0 <= c["total"] <= 1.0, c
        assert 0.0 <= c["confidence"] <= 1.0
        assert set(c["dims"]) == {"driver", "structure", "position", "volume"}
        assert r["factors"], "应生成因子"
        for f in r["factors"]:
            assert 0 <= f.strength <= 1 and 0 <= f.reliability <= 1
            assert f.detail and f.theory and "None" not in str(f.value)
    assert r1["compose"]["total"] > r2["compose"]["total"], \
        "上涨趋势的综合分应高于下跌趋势（%s vs %s）" % (
            r1["compose"]["total"], r2["compose"]["total"])
    print("  四维分数范围、置信度、因子字段均合法")
    print("  上涨趋势 %.2f > 下跌趋势 %.2f ✓" % (r1["compose"]["total"],
                                              r2["compose"]["total"]))

    print("\n=== 5. 边界 ===")
    assert analyze_deep(None)["detail"]
    assert analyze_deep(make_ohlcv(30, seed=1))["detail"]
    assert swing_points(None)["detail"]
    print("  空输入/短序列安全")
    print("\nmarket_phase self-test OK")
