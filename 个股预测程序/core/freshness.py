# -*- coding: utf-8 -*-
"""
通用数据新鲜度校验 —— 覆盖所有外部数据源。

为什么需要通用机制
------------------
已经踩过两次同样的坑：
  · 开户数 ``stock_account_statistics_em`` 停在 2023-08（三年旧）
  · CPI/PPI/M2 停在 2025-09（一年多旧）

每次都是"先给结论、后才发现数据过期"。这种事必须由程序自动兜住，
而不是靠人肉核对。所以把校验抽成通用设施：

  1. **注册表**：每个外部数据源声明自己的频率与允许滞后天数
  2. **validate()**：传日期（或 DataFrame），返回是否新鲜 + 距今多少天
  3. **filter_fresh()**：批量过滤，返回（可用, 过期, 缺日期）
  4. **snapshot()**：全局健康快照，供界面/报告展示

频率与默认上限
--------------
日频 5 天（跳过周末/短假）、周频 14 天、月频 70 天、季频 130 天。
节假日期间"最新交易日"可能落后 3~5 天，所以日频给 5 天而不是 1 天。
"""
from __future__ import annotations

import threading

import pandas as pd

from .utils import get_logger

log = get_logger()

#: 频率 → 允许滞后天数
FREQ_DAYS = {"daily": 5, "weekly": 14, "monthly": 70, "quarterly": 130,
             "frozen": 10 ** 6}

#: 外部数据源注册表：name → (中文名, 频率, 备注)
SOURCES = {
    # 行情
    "daily": ("个股/基金日线", "daily", "腾讯/新浪/东财"),
    "index_history": ("指数历史", "daily", "腾讯/新浪"),
    "board_list": ("行业板块列表", "daily", "东财/新浪"),
    "board_history": ("板块历史K线", "daily", "东财 push2his / 同花顺"),
    "fund_nav": ("基金净值", "daily", "东财 f10/lsjz"),
    # 资金与估值
    "margin": ("两融余额", "daily", "沪深交易所（T+1 披露）"),
    "north": ("北向资金", "daily", "东财（2024-08 起口径变化）"),
    "index_pe": ("指数PE", "daily", "乐咕乐股"),
    "nav_premium": ("折溢价", "daily", "腾讯实时"),
    # 宏观（月频）
    "pmi": ("PMI", "monthly", "统计局/财新"),
    "cpi": ("CPI", "monthly", "统计局"),
    "ppi": ("PPI", "monthly", "统计局"),
    "m2": ("M2", "monthly", "央行"),
    "lpr": ("LPR", "monthly", "央行"),
    "accounts": ("新增开户数", "monthly", "东财/中登"),
    # 上交所口径：沪市A股新开户（万户，含一人多户），与中登「新增投资者」
    # （一码通，按人计）不是同一个数。仅供手动填写，见 macro_source。
    "sse_new_accounts": ("沪市A股新开户", "monthly", "上交所统计月报（手动）"),
    # 财务（季频）
    "financials": ("财务数据", "quarterly", "东财"),
    # ---- 已知上游冻结（2026-10 核实，勿重复调查）----
    #   akshare 1.18.47（当前最新）的 macro_china_* 与
    #   stock_account_statistics_em 返回的历史均止于下列日期，
    #   属**上游数据源停止更新**，升级 akshare 无效。
    #   macro_china_cpi_yearly   → 2025-09-10
    #   macro_china_ppi_yearly   → 2025-09-10
    #   macro_china_m2_yearly    → 2025-09-12
    #   macro_china_pmi_yearly   → 2025-08-31
    #   stock_account_statistics_em → 2023-08-01
    #   东财 datacenter RPT_STOCK_OPEN_DATA（报表名从页面 JS 挖出）
    #   → 同样止于 **2023-08**，与 akshare 同源。
    #   结论：**中登已停止公开发布月度新增投资者数**，不是接口问题。
    #   已接入该源作主源（字段更全：新增/环比/同比/期末投资者总数），
    #   但因数据过期，新鲜度校验会排除它，不参与打分。
    #   处置：这些源一律走新鲜度校验，过期即排除，不用旧数据充数。
}

_LOCK = threading.RLock()
_REGISTRY: dict = {}


def limit_days(name: str, freq: str | None = None) -> int:
    """该数据源允许的最大滞后天数。"""
    if freq:
        return FREQ_DAYS.get(freq, 70)
    rec = SOURCES.get(name)
    if rec:
        return FREQ_DAYS.get(rec[1], 70)
    return 70


def _age_days(dt) -> float | None:
    """距今多少天；无法解析返回 None。"""
    if dt is None or dt == "":
        return None
    try:
        d = pd.to_datetime(str(dt)[:10], errors="coerce")
        if d is None or pd.isna(d):
            return None
        return float((pd.Timestamp.now().normalize() - d).days)
    except Exception:  # noqa: BLE001
        return None


def validate(name: str, dt=None, *, freq: str | None = None,
             limit: int | None = None) -> dict:
    """校验单个数据点/序列的新鲜度。

    Parameters
    ----------
    name : 数据源名（用于查注册表）
    dt   : 日期字符串 / Timestamp / 含日期列的 DataFrame
    freq : 覆盖频率
    limit: 覆盖天数上限

    Returns
    -------
    dict(status, age_days, limit, date, stale, unknown, note)
      status: ``fresh`` 新鲜 / ``stale`` 过期 / ``unknown`` 无日期
    """
    lim = int(limit) if limit is not None else limit_days(name, freq)
    date_val = dt
    if isinstance(dt, pd.DataFrame):
        date_val = None
        for c in dt.columns:
            if any(k in str(c) for k in ("日期", "交易日期", "时间", "date", "month")):
                try:
                    s = pd.to_datetime(dt[c], errors="coerce").dropna()
                    if len(s):
                        date_val = s.max()
                        break
                except Exception:  # noqa: BLE001
                    continue
    age = _age_days(date_val)
    out = {"source": name, "limit": lim, "date": str(date_val)[:10] if date_val else "",
           "age_days": age, "status": "unknown", "stale": False, "unknown": True,
           "note": ""}
    if age is None:
        out["note"] = "取不到日期，无法校验新鲜度（按「不可验证」处理）"
    elif age > lim:
        out.update({"status": "stale", "stale": True, "unknown": False})
        out["note"] = "数据过旧：距今 %.0f 天，超过 %d 天上限" % (age, lim)
    else:
        out.update({"status": "fresh", "unknown": False})
        out["note"] = "新鲜（距今 %.0f 天，上限 %d 天）" % (age, lim)
    with _LOCK:
        _REGISTRY[name] = out
    return out


def filter_fresh(name: str, items: dict, *, freq: str | None = None,
                 limit: int | None = None) -> dict:
    """批量校验 ``{key: date}``，返回分组结果。

    Returns
    -------
    dict(fresh, stale, unknown, detail)
    """
    fresh, stale, unknown = {}, {}, {}
    lim = int(limit) if limit is not None else limit_days(name, freq)
    for k, v in (items or {}).items():
        r = validate("%s.%s" % (name, k), v, limit=lim)
        if r["status"] == "fresh":
            fresh[k] = v
        elif r["status"] == "stale":
            stale[k] = (r["age_days"], lim, r["date"])
        else:
            unknown[k] = v
    bits = []
    if fresh:
        bits.append("可用 %d 项" % len(fresh))
    if stale:
        bits.append("过期排除 %d 项（%s）" % (
            len(stale), "、".join("%s 距今%.0f天" % (k, a)
                                 for k, (a, _l, _d) in stale.items())))
    if unknown:
        bits.append("无日期 %d 项（%s）" % (len(unknown), "、".join(unknown)))
    return {"fresh": fresh, "stale": stale, "unknown": unknown,
            "detail": "；".join(bits)}


def snapshot() -> dict:
    """全局健康快照：所有已校验数据源的当前状态。"""
    with _LOCK:
        items = dict(_REGISTRY)
    bad = {k: v for k, v in items.items() if v["status"] != "fresh"}
    return {"checked": len(items), "bad": len(bad), "items": items,
            "detail": ("已校验 %d 项，其中 %d 项过期/不可验证：%s"
                       % (len(items), len(bad),
                          "、".join("%s" % k for k in list(bad)[:8]))
                       if bad else "已校验 %d 项，全部新鲜" % len(items))}


def reset() -> None:
    with _LOCK:
        _REGISTRY.clear()


if __name__ == "__main__":  # pragma: no cover
    import datetime as _dt
    import warnings

    warnings.filterwarnings("ignore")

    print("=== 1. 单点校验 ===")
    today = _dt.date.today()
    r1 = validate("daily", today)
    print("  今天:", r1["status"], "|", r1["note"])
    assert r1["status"] == "fresh"
    r2 = validate("daily", today - _dt.timedelta(days=30))
    print("  30天前:", r2["status"], "|", r2["note"])
    assert r2["status"] == "stale" and r2["stale"]
    r3 = validate("cpi", today - _dt.timedelta(days=400))
    print("  月频400天:", r3["status"], "| 上限", r3["limit"])
    assert r3["status"] == "stale" and r3["limit"] == 70
    r4 = validate("lpr", "")
    print("  无日期:", r4["status"], "|", r4["note"])
    assert r4["status"] == "unknown" and r4["unknown"]

    print("\n=== 2. 频率默认上限 ===")
    assert limit_days("daily") == 5
    assert limit_days("cpi") == 70
    assert limit_days("financials") == 130
    assert limit_days("unknown_source") == 70
    print("  日频5/月频70/季频130/未知70 ✓")

    print("\n=== 3. DataFrame 取最新日期 ===")
    df = pd.DataFrame({"日期": ["2026-01-01", "2026-10-20", "bad"]})
    r5 = validate("index_pe", df)
    print("  df 最新日期:", r5["date"], "|", r5["status"])
    assert r5["date"] == "2026-10-20"

    print("\n=== 4. 批量过滤 ===")
    items = {"a": today, "b": today - _dt.timedelta(days=400),
             "c": "", "d": today - _dt.timedelta(days=2)}
    got = filter_fresh("macro_batch", items, limit=70)
    print(" ", got["detail"])
    assert set(got["fresh"]) == {"a", "d"}
    assert set(got["stale"]) == {"b"}
    assert set(got["unknown"]) == {"c"}

    print("\n=== 5. 全局快照 ===")
    snap = snapshot()
    print(" ", snap["detail"][:120])
    assert snap["checked"] > 0

    print("\n=== 6. 边界 ===")
    assert validate("x", None)["status"] == "unknown"
    assert filter_fresh("x", {})["detail"] == ""
    reset()
    assert snapshot()["checked"] == 0
    print("  空输入/重置安全")
    print("\nfreshness self-test OK")
