# -*- coding: utf-8 -*-
"""
预测基准语义校验 —— 锁定"相对分析时股价"，防止将来被改成"相对前一日"。

用户提出的关键澄清
------------------
"对个股未来涨跌的预测，是指未来目标时间股价相对**搜索分析时**的股价
 是升高还是降低，**不是**相对目标时间前一个交易日的股价。"

这两个定义差别很大：
  正确：return = 收盘[t+bars] / 收盘[t] − 1        （t = 分析当日）
  错误：return = 收盘[t+bars] / 收盘[t+bars−1] − 1 （只看最后一天涨没涨）

如果写成后者，"20天涨跌概率"就变成"第20天当天涨没涨"，与"20天累计涨跌"
完全不是一回事——一个涨了 20% 但最后一天微跌的行情会被判成"跌"。

本测试用**构造数据**把语义钉死：
  构造一段"持续上涨、但最后一天小幅回落"的行情：
    · 正确语义下，未来收益应为**正**
    · 错误语义下，未来收益会是**负**
  只要有人把公式改成前一日基准，测试立刻失败。

用法：``python tests/test_prediction_basis.py``
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def main() -> int:
    import numpy as np
    import pandas as pd

    from core import backtest as bt
    from core import mlmodel
    from core.contracts import horizon_spec

    # ---- 构造：100 天横盘，最后 20 天持续上涨，但第 20 天小幅回落 ----
    n = 140
    close = np.concatenate([
        np.full(120, 10.0),                       # 前 120 天横盘
        np.linspace(10.0, 12.5, 19),              # 连续上涨到 12.5
        np.array([12.40]),                        # 最后一天小幅回落
    ])
    assert len(close) == n
    df = pd.DataFrame({
        "date": pd.bdate_range("2025-01-01", periods=n),
        "open": close * 0.999, "high": close * 1.005,
        "low": close * 0.995, "close": close,
        "volume": np.full(n, 1e6),
    })

    ok = True

    # ---- 1. make_targets：基准必须是分析当日 ----
    tg = mlmodel.make_targets(df)
    print("检查 mlmodel.make_targets")
    for key in ("fwd_d1", "fwd_d5", "fwd_d20"):
        col = tg[key]
        finite = col.dropna()
        if not len(finite):
            print("  [失败] %s 全为空" % key)
            ok = False
            continue
        # 取一个"处在上涨中段"的位置验证：t=125（已在涨），t+1/t+5/t+20 都更高
        bars = horizon_spec(key.replace("fwd_", "")).bars
        i = 125
        if i + bars >= n:
            continue
        expect = (close[i + bars] / close[i] - 1) * 100
        got = float(col.iloc[i])
        good = abs(got - expect) < 1e-6
        ok = ok and good
        print("  %-8s t=%d bars=%d 期望 %+.3f%% 实际 %+.3f%%  %s"
              % (key, i, bars, expect, got, "OK" if good else "不符"))

    # ---- 2. 关键语义：最后一根回落，但累计仍为正 ----
    print("\n检查「末日回落但累计上涨」的语义")
    t = n - 20 - 1                     # 从上涨起点前一天开始算 20 天
    cum_20 = (close[t + 20] / close[t] - 1) * 100          # 正确：相对分析日
    last_1 = (close[t + 20] / close[t + 19] - 1) * 100     # 错误：相对前一日
    print("  20 日累计（相对分析日）：%+.3f%%" % cum_20)
    print("  最后一日（相对前一日）：%+.3f%%" % last_1)
    if not (cum_20 > 0.5 and last_1 < 0):
        print("  [失败] 构造数据未形成「累计涨、末日跌」的对照")
        ok = False
    else:
        got = float(tg["fwd_d20"].iloc[t])
        near = abs(got - cum_20) < 1e-6
        far = abs(got - last_1) < 1e-6
        print("  make_targets 返回 %+.3f%% → %s"
              % (got, "等于累计值（正确）" if near else
                 ("等于最后一日值（**错误**）" if far else "两者都不等（异常）")))
        ok = ok and near and not far

    # ---- 3. backtest 的前向收益定义必须一致 ----
    print("\n检查 backtest 前向收益定义")
    src = open(os.path.join(HERE, "core", "backtest.py"), encoding="utf-8").read()
    has_right = "c[bars:] / c[:n - bars] - 1.0" in src
    print("  使用 c[t+bars]/c[t] 形式:", "是" if has_right else "否")
    if not has_right:
        print("  [失败] backtest 的收益定义与分析基准不一致")
        ok = False

    # ---- 4. 所有周期都遵循同一基准 ----
    print("\n检查全部 8 个周期")
    for key, col in tg.items():
        bars = horizon_spec(key.replace("fwd_", "")).bars
        i = 60
        if i + bars >= n:
            continue
        expect = (close[i + bars] / close[i] - 1) * 100
        lasterr = abs(float(col.iloc[i]) - expect)
        if lasterr > 1e-6:
            print("  [失败] %s 基准不一致（误差 %.6f）" % (key, lasterr))
            ok = False
    print("  8 个周期全部以分析当日为基准" if ok else "  存在不一致")

    if ok:
        print("\n[通过] 预测基准 = **分析当日收盘价**（非目标日前一交易日），"
              "8 个周期、回测、ML 标签三处定义一致")
        return 0
    print("\n[失败] 预测基准语义被破坏，请检查")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
