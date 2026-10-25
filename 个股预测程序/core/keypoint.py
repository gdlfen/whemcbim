# -*- coding: utf-8 -*-
"""
关键点位的动态加权 —— 走势反转/突破的近几天，当天量能与资金比累计值更重要。

为什么需要
----------
因子汇总默认把"今日主力净额"和"近20日累计主力净额"当成并列的证据。
但在**关键点位**附近，这个假设是错的：

    股价放量突破年线的那一天，今天这一根量能的信息量
    **远大于**前面 20 天的累计量能——因为前面 20 天是旧状态，
    今天这根才代表状态切换。

反过来，在长期横盘、无事件驱动的平淡期，单日数据的噪声很大，
反而应该更看重多日累计的稳定性。

所以本模块做两件事：
1. **识别关键点**（突破/跌破关键位、放量异动、跳空、大涨大跌、量价背离）
2. **按"离关键点多近"动态调权重**：
     · 关键点附近 → 抬升"当日/最新"类因子，压低"累计N日"类因子
     · 平淡期     → 反向微调（累计更可靠）
  同时把整体权重总和归一，避免"调权即放杠杆"。

权重调整幅度上限
----------------
`MAX_BOOST` / `MAX_DAMP` 保证不会把某一类因子放到压倒性地位
（否则单一数据源出错就会主导结论）。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .contracts import Category, DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Factor
from .utils import get_logger

log = get_logger()

#: 单因子权重乘子的上下限（防止某一类因子压倒性主导）
MAX_BOOST = 1.60
MAX_DAMP = 0.62

#: "当日/最新"类因子：id 里带这些词的，在关键点附近加权
RECENT_PATTERNS = (
    r"today", r"_last$", r"_1d$", r"latest", r"spike", r"surge",
    r"breakout", r"breakdown", r"break_", r"gap", r"limit_up", r"limit_down",
    r"big_move", r"turnover", r"volume_ratio", r"vol_ratio", r"intraday",
    r"amplitude", r"reversal", r"key_level", r"support", r"resistance",
)

#: "累计N日"类因子：在关键点附近降权
CUMULATIVE_PATTERNS = (
    r"_5d$", r"_10d$", r"_20d$", r"_60d$", r"_120d$", r"cum", r"_sum",
    r"_avg", r"_ma\d+", r"trend_\d+", r"median", r"percentile",
)


def _match(pat_list, text: str) -> bool:
    return any(re.search(p, text, re.I) for p in pat_list)


# --------------------------------------------------------------------------- #
# 一、关键点识别
# --------------------------------------------------------------------------- #
def detect_key_events(df: pd.DataFrame, lookback: int = 250) -> dict:
    """识别最近的关键点事件，并给出"当前离关键点多近"。

    Returns
    -------
    dict：``focus``(0~1) / ``events``(列表) / ``near_level`` / ``detail``
    """
    out: dict = {"focus": 0.0, "events": [], "detail": "",
                 "near_level": "", "breakout": "", "volume_anomaly": False}
    if df is None or not len(df) or "close" not in df.columns:
        out["detail"] = "缺少行情数据，关键点检测跳过"
        return out
    d = df.dropna(subset=["close"]).copy()
    c = d["close"].astype(float).to_numpy()
    n = len(c)
    if n < 70:
        out["detail"] = "样本不足 70 根，关键点检测跳过"
        return out
    v = (d["volume"].astype(float).to_numpy()
         if "volume" in d.columns else np.full(n, np.nan))
    cur = float(c[-1])
    events: list = []

    # ---- 1) 均线关键位（年线/半年线/季线）的突破或跌破 ----
    for bars, name in ((250, "年线"), (120, "半年线"), (60, "季线"), (20, "月线")):
        if n < bars + 6:
            continue
        ma = float(np.mean(c[-bars:]))
        ma_prev = float(np.mean(c[-bars - 1:-1]))
        if not np.isfinite(ma) or ma <= 0:
            continue
        above_now, above_prev = cur >= ma, float(c[-2]) >= ma_prev
        gap_pct = (cur / ma - 1) * 100
        if above_now and not above_prev:
            events.append({"type": "breakout", "level": name,
                           "side": "up", "days_ago": 0,
                           "desc": "今日站上%s（%.2f）" % (name, ma)})
        elif (not above_now) and above_prev:
            events.append({"type": "breakdown", "level": name,
                           "side": "down", "days_ago": 0,
                           "desc": "今日跌破%s（%.2f）" % (name, ma)})
        elif abs(gap_pct) <= 0.8:
            # 贴着均线：也是关键决策位。只保留最贴近的一条，
            # 否则多条均线同时贴近会累积成过高聚焦度（平淡期误报）
            if not any(e["type"] == "testing" for e in events):
                events.append({"type": "testing", "level": name,
                               "side": "up" if above_now else "down",
                               "days_ago": 0,
                               "desc": "股价贴近%s（偏离 %+.2f%%），处于方向选择位"
                                       % (name, gap_pct)})

    # ---- 2) 前高/前低（箱体上下沿）----
    win = min(n, lookback)
    prior = c[-win:-5] if n > win + 5 else c[:-5]
    if len(prior) >= 20:
        hi, lo = float(np.max(prior)), float(np.min(prior))
        # 区间振幅不足 3% 说明是窄幅横盘，此时"突破前高"没有意义
        # （微小波动也会被算成突破，导致平淡期误报）
        rng_ok = lo > 0 and (hi / lo - 1) >= 0.03
        if not rng_ok:
            prior = np.array([])
        # 近 5 日内是否突破前高 / 跌破前低
        rec5 = c[-5:]
        if np.max(rec5) >= hi * 0.999:
            days = int(len(rec5) - 1 - int(np.argmax(rec5 >= hi * 0.999)))
            events.append({"type": "breakout", "level": "前高",
                           "side": "up", "days_ago": max(days, 0),
                           "desc": "%d 日前突破近 %d 日高点 %.2f"
                                   % (max(days, 0), win, hi)})
        if np.min(rec5) <= lo * 1.001:
            days = int(len(rec5) - 1 - int(np.argmin(rec5 <= lo * 1.001)))
            events.append({"type": "breakdown", "level": "前低",
                           "side": "down", "days_ago": max(days, 0),
                           "desc": "%d 日前跌破近 %d 日低点 %.2f"
                                   % (max(days, 0), win, lo)})

    # ---- 3) 量能异动（今日量 / 20日均量）----
    if np.isfinite(v).sum() >= 25:
        v20 = float(np.nanmean(v[-21:-1]))
        v_now = float(v[-1])
        if v20 > 0 and np.isfinite(v_now):
            vr = v_now / v20
            if vr >= 1.8:
                events.append({"type": "volume_spike", "level": "量能",
                               "side": "up" if c[-1] >= c[-2] else "down",
                               "days_ago": 0,
                               "desc": "今日量能是 20 日均量的 %.1f 倍" % vr})
                out["volume_anomaly"] = True

    # ---- 4) 单日大涨大跌（±5%）----
    if n >= 2 and c[-2] > 0:
        chg = (c[-1] / c[-2] - 1) * 100
        if abs(chg) >= 5.0:
            events.append({"type": "big_move", "level": "单日波动",
                           "side": "up" if chg > 0 else "down", "days_ago": 0,
                           "desc": "今日%s %.2f%%" % ("大涨" if chg > 0 else "大跌",
                                                     abs(chg))})

    # ---- 5) 跳空缺口 ----
    if all(k in d.columns for k in ("open", "high", "low")) and n >= 3:
        o = d["open"].astype(float).to_numpy()
        h = d["high"].astype(float).to_numpy()
        lo_ = d["low"].astype(float).to_numpy()
        for back in (0, 1, 2):
            i = n - 1 - back
            if i < 1:
                break
            if lo_[i] > h[i - 1] * 1.001:
                events.append({"type": "gap", "level": "跳空",
                               "side": "up", "days_ago": back,
                               "desc": "%d 日前向上跳空" % back})
                break
            if h[i] < lo_[i - 1] * 0.999:
                events.append({"type": "gap", "level": "跳空",
                               "side": "down", "days_ago": back,
                               "desc": "%d 日前向下跳空" % back})
                break

    # ---- 综合"聚焦度"：越近、越多事件 → focus 越高 ----
    if not events:
        out["detail"] = "近期无关键点事件（处于常态区间），按常规权重处理"
        return out
    score = 0.0
    for e in events:
        w = {0: 1.0, 1: 0.75, 2: 0.55}.get(int(e.get("days_ago") or 0), 0.35)
        if e["type"] in ("breakout", "breakdown"):
            w *= 1.35
        elif e["type"] == "volume_spike":
            w *= 1.15
        elif e["type"] == "testing":
            w *= 0.5
        score += w
    out["focus"] = round(min(score / 2.6, 1.0), 3)
    out["events"] = events[:8]
    # 最近的突破方向，供文案使用
    brk = [e for e in events if e["type"] in ("breakout", "breakdown")]
    if brk:
        e0 = min(brk, key=lambda x: int(x.get("days_ago") or 0))
        out["breakout"] = "%s（%s）" % (e0["desc"], "向上" if e0["side"] == "up" else "向下")
    lv = [e for e in events if e["type"] == "testing"]
    if lv:
        out["near_level"] = lv[0]["desc"]
    out["detail"] = ("识别到 %d 个关键点事件，聚焦度 %.0f%%：%s"
                     % (len(events), out["focus"] * 100,
                        "；".join(e["desc"] for e in events[:4])))
    return out


# --------------------------------------------------------------------------- #
# 二、按聚焦度动态调权
# --------------------------------------------------------------------------- #
def reweight_factors(factors, key: dict) -> dict:
    """按关键点聚焦度调整因子的 weight_multiplier（就地修改）。

    · focus 越高（越接近突破/异动）→ "当日/最新"类因子越被抬升，
      "累计N日"类因子越被压低
    · focus 低（常态）→ 反向微调：累计数据更稳健
    · 调完把**总权重归一**，保证"调权不改总杠杆"

    Returns
    -------
    dict：``{"focus","boosted","damped","note"}``
    """
    stat = {"focus": float(key.get("focus") or 0.0), "boosted": 0,
            "damped": 0, "note": ""}
    if not factors:
        return stat
    focus = stat["focus"]
    if focus <= 0.02:
        stat["note"] = "无关键点事件，权重未调整"
        return stat
    # 关键点附近：抬"当日"、压"累计"；平淡期（focus 很小）方向相反
    boost = 1.0 + (MAX_BOOST - 1.0) * min(focus, 1.0)
    damp = 1.0 - (1.0 - MAX_DAMP) * min(focus, 1.0)
    try:
        before = []
        for f in factors:
            w0 = float(getattr(f, "weight_multiplier", 1.0) or 1.0)
            before.append(w0)
            fid = str(getattr(f, "id", "") or "")
            is_recent = _match(RECENT_PATTERNS, fid)
            is_cum = _match(CUMULATIVE_PATTERNS, fid)
            if is_recent and not is_cum:
                f.weight_multiplier = w0 * boost
                stat["boosted"] += 1
            elif is_cum and not is_recent:
                f.weight_multiplier = w0 * damp
                stat["damped"] += 1
        # 归一：保持总权重不变（避免"调权 = 放杠杆"）
        after = [float(getattr(f, "weight_multiplier", 1.0) or 1.0)
                 for f in factors]
        sb, sa = float(sum(before)), float(sum(after))
        if sb > 0 and sa > 0 and abs(sa - sb) / sb > 1e-6:
            k = sb / sa
            for f in factors:
                f.weight_multiplier = float(
                    getattr(f, "weight_multiplier", 1.0) or 1.0) * k
        stat["note"] = ("关键点加权：当日/最新类因子 ×%.2f（%d 个），"
                        "累计N日类 ×%.2f（%d 个），总权重已归一"
                        % (boost, stat["boosted"], damp, stat["damped"]))
    except Exception as exc:  # noqa: BLE001
        log.debug("关键点调权失败: %s", exc)
        stat["note"] = "关键点调权失败，使用默认权重"
    return stat


# --------------------------------------------------------------------------- #
# 三、因子
# --------------------------------------------------------------------------- #
def keypoint_factors(df: pd.DataFrame, key: dict | None = None) -> list[Factor]:
    """把关键点事件转成因子（让用户知道"现在为什么特殊"）。"""
    out: list[Factor] = []
    key = key or detect_key_events(df)
    focus = float(key.get("focus") or 0.0)
    if focus <= 0.05:
        return out
    events = key.get("events") or []
    # 突破 / 跌破
    brk = [e for e in events if e["type"] in ("breakout", "breakdown")]
    if brk:
        e0 = min(brk, key=lambda x: int(x.get("days_ago") or 0))
        up = (e0.get("side") == "up")
        out.append(Factor(
            id="keypoint_breakout", name="关键位%s：%s"
                 % ("突破" if up else "跌破", e0.get("level") or ""),
            category=Category.TECH,
            direction=DIR_BULL if up else DIR_BEAR,
            strength=min(0.45 + focus * 0.45, 0.9),
            reliability=0.70, value=e0.get("desc") or "",
            detail=("%s。关键位突破/跌破是**状态切换**信号，"
                    "此时当日量能与资金流的权重已被自动提高，"
                    "累计多日数据的权重相应降低（因为旧状态的信息在贬值）。"
                    % (e0.get("desc") or "")),
            theory="关键位突破/趋势状态切换",
            horizon=("d1", "d5", "d10", "d20")))
    # 量能异动
    vs = [e for e in events if e["type"] == "volume_spike"]
    if vs:
        e0 = vs[0]
        out.append(Factor(
            id="keypoint_volume_spike", name="量能异动",
            category=Category.TECH,
            direction=DIR_BULL if e0.get("side") == "up" else DIR_BEAR,
            strength=min(0.5 + focus * 0.35, 0.85), reliability=0.66,
            value=e0.get("desc") or "",
            detail=("%s。放量代表多空分歧加大、资金真金白银表态，"
                    "在关键位附近放量的可信度显著高于平时。"
                    % (e0.get("desc") or "")),
            theory="量价关系/量能异动", horizon=("d1", "d5")))
    # 贴线待变（方向未定）
    test = [e for e in events if e["type"] == "testing"]
    if test and not brk:
        out.append(Factor(
            id="keypoint_testing", name="贴近关键位待变",
            category=Category.TECH, direction=DIR_NEUTRAL,
            strength=0.3, reliability=0.55, value=test[0].get("level") or "",
            detail=("%s。方向尚未选择，此时信号可靠性下降，"
                    "建议等放量突破/跌破确认后再动作。" % (test[0].get("desc") or "")),
            theory="支撑压力/方向选择", horizon=("d1", "d5")))
    # 跳空
    gp = [e for e in events if e["type"] == "gap"]
    if gp:
        e0 = gp[0]
        out.append(Factor(
            id="keypoint_gap", name="跳空缺口",
            category=Category.TECH,
            direction=DIR_BULL if e0.get("side") == "up" else DIR_BEAR,
            strength=0.55, reliability=0.62, value=e0.get("desc") or "",
            detail="%s。缺口往往对应情绪突变，短期惯性强，但也要防回补。"
                   % (e0.get("desc") or ""),
            theory="缺口理论", horizon=("d1", "d5", "d10")))
    return out


# --------------------------------------------------------------------------- #
# 四、一站式
# --------------------------------------------------------------------------- #
def analyze(df: pd.DataFrame) -> dict:
    """检测关键点 → 生成因子 → 返回调权所需的 focus。"""
    key = detect_key_events(df)
    return {"key": key, "factors": keypoint_factors(df, key),
            "detail": key.get("detail") or ""}


if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")

    from .selftest_data import make_ohlcv

    print("=== 1. 突破场景（长期横盘后放量突破年线）===")
    n = 320
    base = np.full(n, 10.0)
    base[:250] = np.linspace(10.5, 9.8, 250)      # 长期下行
    base[250:] = np.linspace(9.9, 10.9, 70)        # 反转上行
    df = make_ohlcv(n, seed=5, start_price=10.0)
    df["close"] = base
    df["high"] = base * 1.01
    df["low"] = base * 0.99
    v = df["volume"].to_numpy(float).copy()
    v[-1] = v[-21:-1].mean() * 2.4                 # 今日放量 2.4 倍
    df["volume"] = v
    r = analyze(df)
    print(" ", r["detail"])
    print("  focus = %.2f | 因子 %d 条" % (r["key"]["focus"], len(r["factors"])))
    for f in r["factors"]:
        print("     %-26s %s" % (f.name[:24], {1: "多", -1: "空", 0: "中"}[f.direction]))
    assert r["key"]["focus"] > 0.3, "突破场景 focus 应较高"
    assert r["factors"], "应生成关键点因子"

    print("\n=== 2. 平淡场景（无事件）===")
    flat = make_ohlcv(300, seed=9, start_price=10.0)
    flat["close"] = 10.0 + np.sin(np.arange(300) / 40) * 0.02
    flat["high"] = flat["close"] * 1.002
    flat["low"] = flat["close"] * 0.998
    r2 = analyze(flat)
    print("  focus = %.2f | 因子 %d 条" % (r2["key"]["focus"], len(r2["factors"])))
    assert r2["key"]["focus"] < 0.5

    print("\n=== 3. 调权效果（当日抬升、累计压低、总权重守恒）===")
    from .contracts import Category as C

    def mk(fid, name):
        return Factor(fid, name, C.TECH, 1, 0.5, 0.6)

    fs = [mk("main_net_last", "今日主力净额"),          # 当日 → 抬
          mk("main_net_20d", "近20日主力净额"),        # 累计 → 压
          mk("volume_ratio", "量比"),                  # 当日 → 抬
          mk("trend_20d", "20日趋势"),                 # 累计 → 压
          mk("macd_cross", "MACD金叉")]                # 中性 → 不动
    before = sum(f.weight_multiplier for f in fs)
    key = {"focus": 0.85}
    st = reweight_factors(fs, key)
    after = sum(f.weight_multiplier for f in fs)
    print("  ", st["note"])
    for f in fs:
        print("     %-22s weight = %.4f" % (f.name, f.weight_multiplier))
    assert fs[0].weight_multiplier > 1.0, "当日类应被抬升"
    assert fs[1].weight_multiplier < 1.0, "累计类应被压低"
    assert abs(fs[4].weight_multiplier - 1.0) < 0.15, "中性因子应基本不变"
    assert abs(before - after) < 1e-6, "总权重必须守恒（防止调权=放杠杆）"
    print("  总权重 %.6f → %.6f（守恒 ✓）" % (before, after))

    print("\n=== 4. 边界 ===")
    assert analyze(None)["key"]["focus"] == 0.0
    assert keypoint_factors(None) == []
    assert reweight_factors([], {})["note"] == ""
    print("  空输入安全")
    print("\nkeypoint self-test OK")
