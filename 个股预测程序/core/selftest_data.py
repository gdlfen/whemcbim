# -*- coding: utf-8 -*-
"""
离线自测数据生成器。

用途：**在没有网络 / 数据源被封禁的环境下，任何一个模块都能独立开发与自测。**

生成的日线数据具有真实市场的一些关键统计特征：
    * 波动率聚集（GARCH 式条件方差）
    * 收益自相关的弱动量与均值回复切换
    * 量价配合（放量伴随大振幅）
    * 阶段性切换：趋势 → 横盘 → 趋势 → 跳空缺口

所有函数都是确定性的（给定 seed 结果一致），便于回归测试。
"""

from __future__ import annotations

import math
import random

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

__all__ = ["make_ohlcv", "make_index", "make_minute_flow", "make_financials_frame"]

_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover"]


def make_ohlcv(n: int = 900, seed: int = 42, start_price: float = 30.0,
               start_date: str = "2022-01-03", symbol: str = "600519",
               regime: str = "mixed", vol_scale: float = 1.0):
    """生成合成日线 OHLCV 表。

    Parameters
    ----------
    n : 交易日根数
    seed : 随机种子（确定性）
    regime : ``"mixed"``（趋势/横盘交替） | ``"up"`` | ``"down"`` | ``"sideways"``

    Returns
    -------
    pandas.DataFrame，列见 :data:`_COLUMNS`（date 为 datetime64，兼容 pandas 2/3）
    """
    import datetime as dt

    import pandas as pd

    rng = random.Random(seed)
    d = dt.date.fromisoformat(start_date)
    dates = []
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += dt.timedelta(days=1)

    price = float(start_price)
    rows = []
    # 阶段计划：每段 40~160 根，趋势强度与方向随机
    seg_left = 0
    drift = 0.0
    vol = 0.016 * vol_scale
    for i in range(n):
        if seg_left <= 0:
            seg_left = rng.randint(45, 160)
            if regime == "up":
                drift, tvol = rng.uniform(0.0006, 0.0022), 0.014
            elif regime == "down":
                drift, tvol = -rng.uniform(0.0006, 0.0022), 0.015
            elif regime == "sideways":
                drift, tvol = rng.uniform(-0.0002, 0.0002), 0.011
            else:
                kind = rng.choices(["trend", "flat", "trend"], weights=[0.45, 0.3, 0.25])[0]
                if kind == "trend":
                    drift = rng.uniform(0.0008, 0.0026) * rng.choice([1, 1, -1])
                    tvol = rng.uniform(0.013, 0.019)
                else:
                    drift, tvol = rng.uniform(-0.0003, 0.0003), rng.uniform(0.008, 0.013)
            drift *= vol_scale
        seg_left -= 1
        # 波动率聚集：向目标波动缓慢回归
        vol = 0.94 * vol + 0.06 * tvol
        shock = rng.gauss(0, 1)
        ret = drift + vol * shock
        if i > 20 and rng.random() < 0.012:          # 偶发跳空
            ret += rng.choice([1, -1]) * rng.uniform(0.03, 0.075)
        ret = max(min(ret, 0.098), -0.098)
        prev_close = price
        open_p = prev_close * (1 + rng.gauss(0, vol * 0.35))
        close_p = prev_close * (1 + ret)
        hi = max(open_p, close_p) * (1 + abs(rng.gauss(0, vol * 0.55)))
        lo = min(open_p, close_p) * (1 - abs(rng.gauss(0, vol * 0.55)))
        # 量价配合：振幅越大成交量越大
        amp = (hi - lo) / max(prev_close, 1e-9)
        base_vol = 2_600_000 * (1 + 6.5 * amp) * rng.uniform(0.72, 1.32)
        volume = int(base_vol)
        amount = volume * (hi + lo + close_p) / 3.0
        turnover = round(volume / 1.25e9 * 100, 4)
        rows.append({
            "date": pd.Timestamp(dates[i]),
            "open": round(open_p, 2), "high": round(hi, 2),
            "low": round(lo, 2), "close": round(close_p, 2),
            "volume": volume, "amount": round(amount, 2),
            "pct_chg": round((close_p / prev_close - 1) * 100, 4),
            "turnover": turnover,
        })
        price = close_p

    df = pd.DataFrame(rows, columns=_COLUMNS)
    df.attrs["symbol"] = symbol
    df.attrs["synthetic"] = True
    return df


def make_index(n: int = 900, seed: int = 7, start_price: float = 3100.0,
               start_date: str = "2022-01-03"):
    """生成合成大盘指数日线（与 :func:`make_ohlcv` 同结构）。"""
    return make_ohlcv(n=n, seed=seed, start_price=start_price, start_date=start_date,
                      symbol="sh000001", regime="mixed", vol_scale=0.45)


def make_minute_flow(n: int = 240, seed: int = 11) -> list:
    """生成合成分时资金流（用于主力资金/大单占比计算）。

    Returns
    -------
    list[dict]，键为 ``time, main_net, super_net, big_net, retail_net, price``
    """
    rng = random.Random(seed)
    out = []
    px = 30.0
    for i in range(n):
        px *= 1 + rng.gauss(0, 0.0015)
        main = rng.gauss(0, 1.0) * 3.5e5
        out.append({
            "time": "%02d:%02d" % (9 + (i * 4) // 60, (i * 4) % 60),
            "main_net": round(main, 1),
            "super_net": round(main * rng.uniform(0.45, 0.75), 1),
            "big_net": round(main * rng.uniform(0.25, 0.55), 1),
            "retail_net": round(-main * rng.uniform(0.6, 1.1), 1),
            "price": round(px, 3),
        })
    return out


def make_financials_frame(n_periods: int = 12, seed: int = 21):
    """生成合成财务摘要表（营收/净利/ROE/负债率等按季度）。

    Returns
    -------
    pandas.DataFrame，列为 ``报告期, 营业收入, 营收同比, 净利润, 净利同比, ROE,
    毛利率, 净利率, 资产负债率, 经营现金流, 每股收益``
    """
    import pandas as pd

    rng = random.Random(seed)
    rows = []
    rev = 3.0e9
    for i in range(n_periods):
        rev *= 1 + rng.uniform(-0.05, 0.22)
        margin = rng.uniform(0.10, 0.32)
        profit = rev * margin
        rows.append({
            "报告期": "202%dQ%d" % (i // 4, i % 4 + 1),
            "营业收入": round(rev, 2),
            "营收同比": round(rng.uniform(-15, 45), 2),
            "净利润": round(profit, 2),
            "净利同比": round(rng.uniform(-25, 60), 2),
            "ROE": round(rng.uniform(4, 24), 2),
            "毛利率": round(rng.uniform(18, 62), 2),
            "净利率": round(margin * 100, 2),
            "资产负债率": round(rng.uniform(25, 70), 2),
            "经营现金流": round(profit * rng.uniform(0.4, 1.6), 2),
            "每股收益": round(rng.uniform(0.15, 2.4), 3),
        })
    return pd.DataFrame(rows)


def _self_test() -> None:
    df = make_ohlcv(300)
    assert set(_COLUMNS).issubset(df.columns), df.columns
    assert len(df) == 300
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-6).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-6).all()
    idx = make_index(120)
    assert len(idx) == 120
    fin = make_financials_frame()
    assert "ROE" in fin.columns
    print("selftest_data OK: ohlcv=%s index=%s fin=%s" % (df.shape, idx.shape, fin.shape))
    print(df.tail(2).to_string())


if __name__ == "__main__":  # pragma: no cover
    _self_test()
