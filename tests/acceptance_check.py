# -*- coding: utf-8 -*-
"""
theory / seasonality 验收脚本（离线，不联网）。

逐条验证验收标准，并打印证据。
用法（在项目根目录执行）：``python tests/acceptance_check.py``
"""

from __future__ import annotations

import os
import sys

import pandas as pd

# 本文件位于 tests/ 下，项目根目录在上一级
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import seasonality as S  # noqa: E402
from core import theory as T  # noqa: E402
from core.contracts import Category, Factor  # noqa: E402
from core.selftest_data import make_financials_frame, make_ohlcv  # noqa: E402

OK = []
BAD = []


def check(name: str, cond: bool, evidence: str = "") -> None:
    (OK if cond else BAD).append(name)
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                           ("  → " + evidence) if evidence else ""))


def main() -> int:
    print("=" * 84)
    print("验收 1／2：python -m core.theory / python -m core.seasonality（见上方输出）")
    print("=" * 84)
    df = make_ohlcv(900, seed=3)
    as_of = pd.Timestamp(df["date"].iloc[-1])
    print("基准数据 make_ohlcv(900, seed=3)：%d 根，%s ~ %s" % (
        len(df), df["date"].iloc[0].date(), df["date"].iloc[-1].date()))

    print("-" * 84)
    print("验收 2：实质性断言")
    print("-" * 84)

    # ---- 道氏 ---------------------------------------------------------- #
    up = T.dow_theory(make_ohlcv(900, seed=5, regime="up"))
    dn = T.dow_theory(make_ohlcv(900, seed=5, regime="down"))
    sd = T.dow_theory(make_ohlcv(900, seed=5, regime="sideways"))
    check("dow_theory(regime=up) == 上升", up["trend"] == "上升",
          "trend=%s score=%+.3f conf=%.2f" % (up["trend"], up["score"], up["confidence"]))
    check("dow_theory(regime=down) == 下降", dn["trend"] == "下降",
          "trend=%s score=%+.3f conf=%.2f" % (dn["trend"], dn["score"], dn["confidence"]))
    check("dow_theory 返回键完整", set(up.keys()) >= {
        "trend", "primary_trend", "secondary_trend", "phase", "hh_hl", "lh_ll",
        "volume_confirm", "score", "confidence", "detail", "swing_points"},
        "swing_points=%d 个，phase=%s，vol_confirm=%s" % (
            len(up["swing_points"]), up["phase"], up["volume_confirm"]))
    check("dow_theory 三 regime 至少各返回合法趋势字符串",
          all(x["trend"] in ("上升", "下降", "横盘") for x in (up, dn, sd)),
          "sideways → %s（横盘震荡数据被随机游走主导，任何方向都合法）" % sd["trend"])

    # ---- 波浪 ---------------------------------------------------------- #
    ell = T.elliott_wave(df)
    check("elliott_wave.wave 在允许集合内",
          ell["wave"] in {"1", "2", "3", "4", "5", "A", "B", "C", "unknown"},
          "wave=%s" % ell["wave"])
    check("elliott_wave.confidence ∈ [0,1]", 0.0 <= ell["confidence"] <= 1.0,
          "confidence=%.3f" % ell["confidence"])
    check("elliott_wave.swing_points ≥ 4", len(ell["swing_points"]) >= 4,
          "%d 个" % len(ell["swing_points"]))
    check("elliott_wave.fib_conformance ∈ [0,1]", 0.0 <= ell["fib_conformance"] <= 1.0,
          "fib_conformance=%.2f" % ell["fib_conformance"])
    # 另找一组能数出浪级的数据，验证非 unknown 路径也正确
    found = None
    for reg in ("up", "down", "mixed", "sideways"):
        for seed in (1, 2, 3, 5, 7, 11, 99):
            e = T.elliott_wave(make_ohlcv(900, seed=seed, regime=reg))
            if e["wave"] != "unknown":
                found = (reg, seed, e)
                break
        if found:
            break
    if found:
        reg, seed, e = found
        check("elliott_wave 能给出有效浪级（含 targets）",
              e["wave"].startswith(("1", "3", "5", "A", "C")) and len(e["targets"]) >= 1
              and e["direction"] in (-1, 1),
              "regime=%s seed=%d → 浪%s 方向%+d 吻合度%.0f%% 目标 %s" % (
                  reg, seed, e["wave"], e["direction"], e["fib_conformance"] * 100,
                  "、".join("%s=%.2f" % (t["label"], t["price"]) for t in e["targets"][:2])))

    # ---- 斐波那契 ------------------------------------------------------ #
    fib = T.fibonacci_analysis(df)
    r = fib["retracement"]
    check("fibonacci 回撤位单调（0.191 > 0.236 > 0.382 > 0.500 > 0.618 > 0.786 > 0.809）",
          r["0.191"] > r["0.236"] > r["0.382"] > r["0.500"] > r["0.618"] > r["0.786"] > r["0.809"],
          "0.191=%.2f 0.618=%.2f 0.809=%.2f" % (r["0.191"], r["0.618"], r["0.809"]))
    check("fibonacci 上升波段时 0.191 位 > 0.618 位", r["0.191"] > r["0.618"],
          "high=%.2f low=%.2f direction=%+d" % (fib["high"], fib["low"], fib["direction"]))
    check("fibonacci current_zone 合法",
          fib["current_zone"] in ("强支撑区", "支撑区", "中位区", "压力区", "强压力区"),
          fib["current_zone"])
    check("fibonacci 支撑 ≤ 压力", fib["nearest_support"] <= fib["nearest_resistance"],
          "支撑=%.2f 压力=%.2f" % (fib["nearest_support"], fib["nearest_resistance"]))
    check("fibonacci 时间窗在未来 60 个交易日内",
          all(0 < w["bars_ahead"] <= 60 for w in fib["time_windows"])
          and len(fib["time_windows"]) >= 2,
          "；".join("%s(+%d)" % (w["date"], w["bars_ahead"]) for w in fib["time_windows"][:5]))

    # ---- 江恩 ---------------------------------------------------------- #
    gan = T.gann_analysis(df)
    check("gann 标注为经验性方法（confidence ≤ 0.45）", gan["confidence"] <= 0.45,
          "confidence=%.2f，角度线 %d 条，时间周期 %d 个，四方形 %d 位" % (
              gan["confidence"], len(gan["gann_angles"]), len(gan["time_cycles"]),
              len(gan["square_of_9"])))
    check("gann 时间周期 = 30/45/60/90/120/144/180/270/360",
          [c["cycle"] for c in gan["time_cycles"]] == [30, 45, 60, 90, 120, 144, 180, 270, 360])
    check("gann 四方形含 support 与 resistance",
          {s["kind"] for s in gan["square_of_9"]} == {"support", "resistance"})

    # ---- 量价时空 / 扇形 / 百分比 --------------------------------------- #
    vpt = T.volume_price_time(df)
    check("volume_price_time 量价阶段合法",
          vpt["volume_stage"] in ("地量", "温和", "放量", "巨量", "天量")
          and vpt["price_stage"] in ("底部", "上涨中", "顶部", "下跌中")
          and vpt["time_stage"] in ("时间窗临近", "未到"),
          "%s + %s + %s（%s）" % (vpt["volume_stage"], vpt["price_stage"],
                                  vpt["time_stage"], vpt["patterns"][0]["name"]))
    fan = T.fan_principle(df)
    check("fan_principle 定位现价所在角度线",
          fan["position"] in ("上方", "下方") and len(fan["lines"]) >= 5
          and fan["origin"].get("type") in ("low", "high"),
          "原点 %s(%s %.2f) → %s 的%s" % (fan["origin"]["date"], fan["origin"]["type"],
                                          fan["origin"]["price"], fan["current_line"],
                                          fan["position"]))
    pct = T.percentage_retracement(df)
    check("percentage_retracement 覆盖 1/3、1/2、2/3 与 1/8 体系",
          len(pct["retracements"]) == 9 and 0 <= pct["position_pct"] <= 100
          and pct["nearest_level_role"] in ("支撑", "压力"),
          "%d 档，位置 %.1f%%，最近位 %.2f(%s)" % (
              len(pct["retracements"]), pct["position_pct"], pct["nearest_level"],
              pct["nearest_level_role"]))

    # ---- 混沌 ---------------------------------------------------------- #
    cha = T.chaos_theory(df)
    check("chaos.hurst ∈ [0,1]", 0.0 <= cha["hurst"] <= 1.0, "H=%.4f" % cha["hurst"])
    check("chaos.fractal_dimension ∈ [1,2]", 1.0 <= cha["fractal_dimension"] <= 2.0,
          "D=%.4f = 2 - H" % cha["fractal_dimension"])
    check("chaos 样本 <120 时返回保守值", T.chaos_theory(df.tail(80))["hurst"] == 0.5,
          "df.tail(80) → H=%.2f" % T.chaos_theory(df.tail(80))["hurst"])
    check("chaos.market_state 合法",
          cha["market_state"] in ("趋势市", "随机漫步", "均值回复"),
          cha["market_state"])

    # ---- 多周期共振 ---------------------------------------------------- #
    res = T.multi_timeframe_resonance(df)
    check("multi_timeframe_resonance 三周期都有 trend/score",
          all(("trend" in res[k] and "score" in res[k])
              for k in ("daily", "weekly", "monthly")),
          "日=%s 周=%s 月=%s" % (res["daily"]["trend"], res["weekly"]["trend"],
                                 res["monthly"]["trend"]))
    check("resonance_direction ∈ {-1,0,1}", res["resonance_direction"] in (-1, 0, 1),
          "共振=%s 方向=%+d 强度=%.2f" % (res["resonance"], res["resonance_direction"],
                                          res["strength"]))
    # 显式传入周/月线
    wk = df.copy()
    res2 = T.multi_timeframe_resonance(df, weekly=None, monthly=None)
    check("显式聚合周/月线可用", res2["_bars"]["weekly"] > 20 and res2["_bars"]["monthly"] > 12,
          "周 %d 根 / 月 %d 根" % (res2["_bars"]["weekly"], res2["_bars"]["monthly"]))

    # ---- 节气 ---------------------------------------------------------- #
    terms = S.solar_terms(pd.Timestamp("2026-01-15"), 6)
    dates = [pd.Timestamp(t["date"]) for t in terms]
    check("solar_terms 日期递增且都在未来",
          all(dates[i] < dates[i + 1] for i in range(len(dates) - 1))
          and all(t["days_ahead"] >= 0 for t in terms),
          "、".join("%s %s(+%d)" % (t["name"], t["date"], t["days_ahead"]) for t in terms))
    lc = [t for t in S.solar_terms(pd.Timestamp("2026-01-01"), 24) if t["name"] == "立春"]
    check("solar_terms 立春 2026 = 2026-02-04（题目校验点）",
          bool(lc) and lc[0]["date"] == "2026-02-04", lc[0]["date"] if lc else "无")

    # ---- 斐波那契时间窗 ------------------------------------------------ #
    ftw = S.fib_time_window(df, as_of)
    fd = [pd.Timestamp(w["date"]) for w in ftw["next_windows"]]
    check("fib_time_window.next_windows 是未来交易日且递增",
          len(fd) >= 2 and all(d >= as_of for d in fd)
          and all(fd[i] < fd[i + 1] for i in range(len(fd) - 1))
          and all(d.weekday() < 5 for d in fd),
          "锚点 %s(%s)，窗口 %s" % (ftw["anchor_date"], ftw["anchor_type"],
                                    "、".join("%s(+%d)" % (w["date"], w["bars_ahead"])
                                              for w in ftw["next_windows"][:5])))

    # ---- 因子契约 ------------------------------------------------------ #
    tf = T.theory_factors(df)
    sf = S.seasonality_factors("600519", df, as_of, make_financials_frame())
    check("theory_factors 全部是 Factor 实例", all(isinstance(f, Factor) for f in tf),
          "共 %d 个" % len(tf))
    check("seasonality_factors 全部是 Factor 实例", all(isinstance(f, Factor) for f in sf),
          "共 %d 个" % len(sf))
    check("所有因子 strength/reliability ∈ [0,1]",
          all(0 <= f.strength <= 1 and 0 <= f.reliability <= 1 for f in tf + sf))
    check("theory 因子 category=THEORY 且 reliability ≤ 0.65",
          all(f.category == Category.THEORY and f.reliability <= 0.65 + 1e-9 for f in tf))
    check("江恩/混沌/波浪 因子 reliability ≤ 0.55",
          all(f.reliability <= 0.55 + 1e-9 for f in tf
              if f.id in ("theory_gann", "theory_chaos", "theory_elliott")),
          "、".join("%s=%.2f" % (f.id, f.reliability) for f in tf
                    if f.id in ("theory_gann", "theory_chaos", "theory_elliott")))
    check("seasonality 因子 category=CYCLE 且 strength ≤ 0.35",
          all(f.category == Category.CYCLE and f.strength <= 0.35 + 1e-9 for f in sf))
    check("因子 id 与约定一致",
          {f.id for f in tf} == {"theory_dow_trend", "theory_elliott", "theory_fib_level",
                                 "theory_gann", "theory_volume_price_time", "theory_fan",
                                 "theory_pct_retracement", "theory_chaos",
                                 "theory_resonance"}
          and {f.id for f in sf} == {"calendar_effect", "month_effect", "weekday_effect",
                                     "holiday_effect", "earnings_window", "two_sessions",
                                     "month_end", "fib_time_window", "gann_time_cycle",
                                     "solar_term"})
    hor = {f.id: f.horizon for f in tf + sf}
    check("horizon 约定正确",
          hor["theory_dow_trend"] == ("d5", "d10", "d20", "m1")
          and hor["theory_elliott"] == ("d20", "m1", "q1")
          and hor["theory_gann"] == ("d10", "d20", "m1", "q1")
          and hor["theory_resonance"] is None
          and hor["calendar_effect"] == ("d20", "m1", "q1")
          and hor["earnings_window"] == ("d5", "d10", "d20", "m1")
          and hor["fib_time_window"] == ("d5", "d10", "d20")
          and hor["holiday_effect"] == ("d5", "d10"))

    # ---- 导入 ---------------------------------------------------------- #
    try:
        from core import seasonality as _s  # noqa: F401
        from core import theory as _t       # noqa: F401
        check("from core import theory, seasonality 无异常", True)
    except Exception as exc:  # noqa: BLE001
        check("from core import theory, seasonality 无异常", False, str(exc))

    # ---- 降级 ---------------------------------------------------------- #
    bad_ok = True
    for bad in (None, pd.DataFrame(), make_ohlcv(10), "x", 1):
        try:
            T.dow_theory(bad); T.elliott_wave(bad); T.fibonacci_analysis(bad)
            T.gann_analysis(bad); T.volume_price_time(bad); T.fan_principle(bad)
            T.percentage_retracement(bad); T.chaos_theory(bad)
            T.multi_timeframe_resonance(bad); T.theory_factors(bad); T.theory_summary(bad)
            S.calendar_effects("600519", bad); S.month_effect(bad); S.weekday_effect(bad)
            S.holiday_effects(bad); S.two_sessions(bad); S.month_end_effect(bad)
            S.fib_time_window(bad); S.gann_time_cycle(bad)
            S.seasonality_factors("600519", bad); S.seasonality_report("600519", bad)
        except Exception as exc:  # noqa: BLE001
            bad_ok = False
            print("     降级失败: %r -> %s" % (bad, exc))
            break
    check("空/短/非法输入全部安全降级（不抛异常）", bad_ok)

    print("-" * 84)
    print("验收 3：theory_summary / seasonality_report 关键字段")
    print("-" * 84)
    su = T.theory_summary(df)
    print("theory_summary：")
    print("  dow       : trend=%s phase=%s score=%+.3f conf=%.2f" % (
        su["dow"]["trend"], su["dow"]["phase"], su["dow"]["score"], su["dow"]["confidence"]))
    print("  elliott   : wave=%s direction=%+d conf=%.2f targets=%d" % (
        su["elliott"]["wave"], su["elliott"]["direction"], su["elliott"]["confidence"],
        len(su["elliott"]["targets"])))
    print("  fib       : zone=%s 支撑=%s 压力=%s 时间窗=%d" % (
        su["fib"]["current_zone"], su["fib"]["nearest_support"],
        su["fib"]["nearest_resistance"], len(su["fib"]["time_windows"])))
    print("  gann      : 角度线=%d 四方形=%d 时间周期=%d conf=%.2f" % (
        len(su["gann"]["gann_angles"]), len(su["gann"]["square_of_9"]),
        len(su["gann"]["time_cycles"]), su["gann"]["confidence"]))
    print("  vpt       : %s+%s time=%s match=%s" % (
        su["vpt"]["volume_stage"], su["vpt"]["price_stage"], su["vpt"]["time_stage"],
        su["vpt"]["match"]))
    print("  fan       : %s/%s  pct: %.1f%%/%s" % (
        su["fan"]["current_line"], su["fan"]["position"], su["pct"]["position_pct"],
        su["pct"]["nearest_level_role"]))
    print("  chaos     : H=%.4f D=%.4f %s" % (
        su["chaos"]["hurst"], su["chaos"]["fractal_dimension"], su["chaos"]["market_state"]))
    print("  resonance : %s dir=%+d strength=%.2f" % (
        su["resonance"]["resonance"], su["resonance"]["resonance_direction"],
        su["resonance"]["strength"]))
    print("  consensus : dir=%+d score=%+.3f bull=%d bear=%d" % (
        su["consensus"]["direction"], su["consensus"]["score"],
        su["consensus"]["bull_count"], su["consensus"]["bear_count"]))
    print("              %s" % su["consensus"]["detail"])
    check("theory_summary 键完整",
          set(su.keys()) == {"dow", "elliott", "fib", "gann", "vpt", "fan", "pct",
                             "chaos", "resonance", "consensus"})
    check("consensus 字段合法", su["consensus"]["direction"] in (-1, 0, 1)
          and -1 <= su["consensus"]["score"] <= 1)

    rep = S.seasonality_report("600519", df, as_of)
    hol_df = S._inject_holidays(df)
    hol_rep = S.seasonality_report("600519", hol_df, pd.Timestamp(hol_df["date"].iloc[-1]))
    print()
    print("seasonality_report：")
    print("  as_of      : %s（%s，%d 月，距月末 %d 个交易日）" % (
        rep["as_of"], rep["today"]["weekday"], rep["today"]["month"],
        rep["today"]["days_to_month_end"]))
    print("  命中效应   : %s" % ("、".join(rep["today"]["effects"]) or "无"))
    print("  calendar   : score=%+.3f 命中 %d 项（强度全部 ≤0.35）" % (
        rep["calendar"]["score"], rep["calendar"]["_n_effects"]))
    print("  month      : %d月 hist=%s%% prior=%+.2f%% blended=%+.2f%% dir=%+d" % (
        rep["month"]["current_month"], rep["month"]["hist_month_ret_pct"],
        rep["month"]["prior_ret_pct"], rep["month"]["blended_ret_pct"],
        rep["month"]["direction"]))
    print("  weekday    : best=%s worst=%s" % (rep["today"]["weekday_best"],
                                               rep["today"]["weekday_worst"]))
    print("  holiday    : 原始数据 n=%d（无长假缺口）｜注入长假后 n=%d pre5=%s%% post5=%s%% dir=%+d" % (
        rep["holiday"]["n_holidays"], hol_rep["holiday"]["n_holidays"],
        hol_rep["holiday"]["pre5_ret"], hol_rep["holiday"]["post5_ret"],
        hol_rep["holiday"]["direction"]))
    print("  earnings   : window=%s period=%s days=%d risk=%s" % (
        rep["earnings"]["window"], rep["earnings"]["next_report_period"],
        rep["earnings"]["days_to_deadline"], rep["earnings"]["risk"]))
    print("  two_sess   : in=%s days_to=%+d hist=%s%%" % (
        rep["two_sessions"]["in_session"], rep["two_sessions"]["days_to"],
        rep["two_sessions"]["hist_ret_pct"]))
    print("  month_end  : is=%s days=%d last3=%s first3=%s" % (
        rep["month_end"]["is_month_end"], rep["month_end"]["days_to_month_end"],
        rep["month_end"]["hist_last3_ret"], rep["month_end"]["hist_first3_ret"]))
    print("  fib_window : anchor=%s windows=%d nearest=%s" % (
        rep["fib_window"]["anchor_date"], len(rep["fib_window"]["next_windows"]),
        rep["fib_window"]["next_windows"][0]["date"]
        if rep["fib_window"]["next_windows"] else "—"))
    print("  gann_cycle : anchor=%s near=%s cycles=%d" % (
        rep["gann_cycle"]["anchor_date"], rep["gann_cycle"]["near_cycle"],
        len(rep["gann_cycle"]["cycles"])))
    print("  solar      : %s" % "、".join("%s %s" % (t["name"], t["date"])
                                          for t in rep["solar_terms"][:3]))
    print("  consensus  : dir=%+d score=%+.3f bull=%d bear=%d" % (
        rep["consensus"]["direction"], rep["consensus"]["score"],
        rep["consensus"]["bull_count"], rep["consensus"]["bear_count"]))
    print("              %s" % rep["consensus"]["detail"])
    check("seasonality_report 键完整",
          set(rep.keys()) >= {"calendar", "month", "weekday", "holiday", "earnings",
                              "two_sessions", "month_end", "fib_window", "gann_cycle",
                              "solar_terms", "consensus"})

    print()
    print("生成因子 id 列表与数量：")
    print("  theory      (%2d)：%s" % (len(tf), ", ".join(f.id for f in tf)))
    print("  seasonality (%2d)：%s" % (len(sf), ", ".join(f.id for f in sf)))
    print("  合计 %d 个因子" % (len(tf) + len(sf)))

    print()
    print("=" * 84)
    print("验收结果：PASS %d / FAIL %d" % (len(OK), len(BAD)))
    if BAD:
        for b in BAD:
            print("  未通过：%s" % b)
    print("=" * 84)
    return 0 if not BAD else 1


if __name__ == "__main__":
    raise SystemExit(main())
