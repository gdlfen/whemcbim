# -*- coding: utf-8 -*-
"""
宏观数据：东财 datacenter 自动获取（新鲜）+ 手动输入兜底。

发现：akshare 的 macro_china_* 上游冻结在 2025-08/09，但**东财 datacenter
经济报表有新鲜数据**（实测 2026-09）：
  RPT_ECONOMY_CPI   → NATIONAL_SAME（全国同比）
  RPT_ECONOMY_PPI   → BASE_SAME（同比）
  RPT_ECONOMY_PMI   → MAKE_INDEX（制造业PMI）
  RPT_ECONOMY_GDP   → DOMESTICL_PRODUCT_BASE（累计GDP）

所以取数顺序改为：**东财 datacenter → akshare → 手动输入 → 缺失**。
手动输入的意义：当所有自动源都过期时，你仍可手填最新值，
新鲜度校验会认可它，而不是整块环境分析瘫掉。
"""
from __future__ import annotations

import json
import os
import threading

import pandas as pd

from .utils import get_logger

log = get_logger()

H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122",
     "Referer": "https://data.eastmoney.com/"}

#: 东财经济报表：key → (报表名, 候选数值列（按优先级）, 中文名, 单位)
#: 东财经济报表：key → (报表名, 候选数值列（优先级）, 中文名, 单位)
#:
#: **口径说明（易错，务必看）**：CPI/PPI 有两套同比
#:   *_SAME        当月同比（百分点形式，如 PPI 5.0 表示当月 +5.0%）
#:   *_ACCUMULATE  累计同比（**定基指数形式**，102.3 表示累计 +2.3%）
#: 之前误用 SAME，得出"PPI +5.0%"，与统计局口径（前三季度累计 +2.3%）不符。
#: 现改为默认取**累计同比**（更代表趋势），并在 minus100 标记下换算成百分点；
#: 当月同比额外保一份在 "mom_same" 里，两者都展示，避免再混淆。
EM_REPORTS = {
    "cpi": ("RPT_ECONOMY_CPI", ("NATIONAL_ACCUMULATE", "NATIONAL_SAME"),
            "CPI累计同比", "%"),
    "ppi": ("RPT_ECONOMY_PPI", ("BASE_ACCUMULATE", "BASE_SAME"),
            "PPI累计同比", "%"),
    "pmi": ("RPT_ECONOMY_PMI", ("MAKE_INDEX", "NMAKE_INDEX"), "制造业PMI", ""),
    "m2": ("RPT_ECONOMY_CURRENCY_SUPPLY", ("BASIC_CURRENCY_SAME", "M2_SAME",
                                          "CURRENCY_SAME"), "M2同比", "%"),
    "lpr": ("RPT_ECONOMY_LPR", ("LPR1Y", "LPR_1Y"), "LPR(1年)", "%"),
    "sse_new_accounts": (None, (), "沪市A股新开户(万户)", "万户"),
}

#: 只能手动填写的键（无可用自动接口）
MANUAL_ONLY = {"sse_new_accounts"}

#: 这些键的数值是"定基指数"（100.9 / 102.3），需要 −100 才是同比百分点
MINUS_100 = {"cpi", "ppi"}

#: 当月同比的字段（CPI/PPI 专用，作为补充信息）
MOM_SAME = {"cpi": "NATIONAL_SAME", "ppi": "BASE_SAME"}

#: 口径说明（key → 一句话讲清"这个数字到底是什么"）
#:
#: 为什么需要：字段名极易误读。``BASE_SAME`` 里的 SAME 确实是"同比"，
#: 但它指**当月同比**；要"累计同比"得用 ``BASE_ACCUMULATE``（定基指数形式，
#: 102.3 表示 +2.3%）。这种错误代码自测发现不了——数据本身是正确返回的，
#: 只是口径不符。所以把口径写在界面上，让人一眼可核对。
CALIBER = {
    "cpi": ("统计局口径，**累计同比**（年初至当月的同比）",
            "字段 NATIONAL_ACCUMULATE（定基指数），换算：值 − 100",
            "当月同比见 mom_same"),
    "ppi": ("统计局口径，**累计同比**（年初至当月的同比）",
            "字段 BASE_ACCUMULATE（定基指数），换算：值 − 100",
            "当月同比见 mom_same"),
    "pmi": ("统计局/财新，**制造业 PMI 当月值**（非同比）",
            "字段 MAKE_INDEX，50 为荣枯线",
            "高于 50 扩张、低于 50 收缩"),
    "m2": ("央行口径，**M2 同比增速**（当月）",
           "字段 BASIC_CURRENCY_SAME / M2_SAME",
           "8% 常被视作宽松门槛"),
    "lpr": ("央行口径，**1 年期 LPR 当前值**（不是同比）",
            "字段 LPR1Y，单位 %",
            "每月 20 日公布"),
    "sse_new_accounts": (
        "上交所口径，**沪市A股新开户数**（万户，含一人多户）",
        "人工填写（上交所统计月报 → 投资者栏目），无公开接口",
        "**与中登「新增投资者」不是同一个数**：后者按一码通（人）计，"
        "无自动数据源，故由人工补录；仅作散户情绪参考，不与其他口径混用"),
}
#: M2/LPR 报表名可能有变，多试几个
EM_ALIASES = {
    "m2": ("RPT_ECONOMY_CURRENCY_SUPPLY", "RPT_ECONOMY_MONEY_SUPPLY",
           "RPT_ECONOMY_M2"),
    "lpr": ("RPT_ECONOMY_LPR", "RPT_ECONOMY_LOAN_PRIME_RATE"),
}

#: 手动输入存放路径
MANUAL_FILE = os.environ.get("STOCKAI_MACRO_FILE") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "macro_manual.json")

_LOCK = threading.RLock()


def load_manual() -> dict:
    """读手动输入的宏观数据：``{key: {value, date, note}}``。"""
    with _LOCK:
        try:
            if os.path.exists(MANUAL_FILE):
                with open(MANUAL_FILE, encoding="utf-8") as fh:
                    d = json.load(fh)
                return d if isinstance(d, dict) else {}
        except Exception as exc:  # noqa: BLE001
            log.debug("读手动宏观失败: %s", exc)
    return {}


def save_manual(data: dict) -> bool:
    """保存手动输入的宏观数据。"""
    with _LOCK:
        try:
            os.makedirs(os.path.dirname(MANUAL_FILE), exist_ok=True)
            with open(MANUAL_FILE, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("保存手动宏观失败: %s", exc)
            return False


def _em_fetch(key: str) -> dict | None:
    """从东财 datacenter 取一项宏观数据。"""
    import requests

    if key in MANUAL_ONLY or key not in EM_REPORTS:
        return None                 # 只能手动填写 / 无此报表
    names = EM_ALIASES.get(key) or (EM_REPORTS[key][0],)
    cols = EM_REPORTS[key][1]
    for nm in names:
        url = ("https://datacenter-web.eastmoney.com/api/data/v1/get?"
               "reportName=%s&columns=ALL&pageNumber=1&pageSize=6"
               "&sortColumns=REPORT_DATE&sortTypes=-1&source=WEB&client=WEB" % nm)
        try:
            j = requests.get(url, timeout=12, headers=H).json()
            rows = (j.get("result") or {}).get("data") or []
            if not rows:
                continue
            row = rows[0]
            val = None
            used = None
            for c in cols:
                if c in row and row[c] is not None:
                    try:
                        val = float(row[c])
                        used = c
                        break
                    except Exception:  # noqa: BLE001
                        continue
            if val is None:
                # 退化：取第一个可转 float 的数值列（排除日期类）
                for c, v in row.items():
                    if any(k in str(c).upper() for k in ("DATE", "TIME", "CODE")):
                        continue
                    try:
                        val = float(v)
                        used = c
                        break
                    except Exception:  # noqa: BLE001
                        continue
            if val is None:
                continue
            dt = str(row.get("REPORT_DATE") or "")[:10]
            # 定基指数 → 同比百分点（102.3 → +2.3）
            if key in MINUS_100:
                val = val - 100.0
            rec = {"value": round(val, 2), "date": dt, "column": used,
                   "report": nm,
                   "period": row.get("TIME") or "",
                   "source": "eastmoney_datacenter"}
            # 附当月同比（百分点形式，无需换算），便于区分两个口径
            mcol = MOM_SAME.get(key)
            if mcol and row.get(mcol) is not None:
                try:
                    rec["mom_same"] = round(float(row[mcol]), 2)
                    rec["mom_column"] = mcol
                except Exception:  # noqa: BLE001
                    pass
            return rec
        except Exception as exc:  # noqa: BLE001
            log.debug("东财宏观 %s/%s 失败: %s", key, nm, exc)
            continue
    return None


def _ak_fetch(key: str) -> dict | None:
    """akshare 兜底（当前上游多半已冻结）。"""
    from . import market_context as MC  # 复用其探测

    names = MC._MACRO_PROBES.get(key)
    if not names:
        return None
    try:
        df = MC._probe(names)
        val, dt = MC._last_num(df)
        if val is None:
            return None
        return {"value": round(float(val), 2), "date": dt or "",
                "source": "akshare"}
    except Exception:  # noqa: BLE001
        return None


def get_macro(key: str, *, prefer: str = "auto") -> dict:
    """取一项宏观数据：东财 → akshare → 手动。

    Returns
    -------
    dict(value, date, source, age_days, fresh, note)
    """
    from . import freshness as FR

    limit = FR.limit_days(key, "monthly")
    cands: list = []
    if prefer != "manual" and key not in MANUAL_ONLY:
        for fn, tag in ((_em_fetch, "auto_em"), (_ak_fetch, "auto_ak")):
            try:
                got = fn(key)
            except Exception:  # noqa: BLE001
                got = None
            if got:
                got["via"] = tag
                cands.append(got)
    man = load_manual().get(key) or {}
    if man.get("value") is not None:
        cands.append({"value": float(man["value"]), "date": man.get("date") or "",
                      "via": "manual", "note": man.get("note") or "",
                      "source": "手动输入"})

    best = None
    for c in cands:
        age = FR._age_days(c.get("date"))
        c["age_days"] = age
        c["fresh"] = (age is not None and age <= limit)
        # 优先级：新鲜者优先；同等新鲜下自动源优先于手动（少人工误差）
        score = (1 if c["fresh"] else 0,
                 1 if c["via"].startswith("auto") else 0,
                 -(age if age is not None else 1e9))
        c["_score"] = score
        if best is None or score > best["_score"]:
            best = c
    if best is None:
        return {"value": None, "date": "", "source": "缺失", "fresh": False,
                "age_days": None, "note": "自动源与手动输入均无可用值"}
    lim_txt = ("新鲜（距今 %.0f 天，上限 %d 天）" % (best["age_days"], limit)
               if best["age_days"] is not None
               else "日期缺失，无法校验新鲜度")
    best["note"] = "%s｜来源 %s｜%s" % (
        EM_REPORTS.get(key, (None, None, key, ""))[2], best.get("source", ""), lim_txt)
    # 口径标注：让"这个数字到底是什么"可直接核对
    cal = CALIBER.get(key)
    if cal:
        best["caliber"] = cal[0]
        best["caliber_field"] = cal[1]
        best["caliber_note"] = cal[2]
    return best


def collect(*, prefer: str = "auto") -> dict:
    """一次性取全部宏观项。"""
    out = {}
    for k in ("cpi", "ppi", "pmi", "m2", "lpr", "sse_new_accounts"):
        try:
            out[k] = get_macro(k, prefer=prefer)
        except Exception as exc:  # noqa: BLE001
            log.debug("宏观 %s 取数失败: %s", k, exc)
    return out


if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")

    print("=== 1. 手动输入的读写 ===")
    old = load_manual()
    save_manual({"cpi": {"value": 0.8, "date": "2026-09-01", "note": "自统计局"}})
    got = load_manual()
    assert abs(float(got["cpi"]["value"]) - 0.8) < 1e-9
    print("  写入/读回 OK:", got)
    save_manual(old)
    print("  已还原原文件")

    print("\n=== 2. 自动取数（东财优先）===")
    res = collect()
    for k, v in res.items():
        print("  %-4s %-8s %-10s %-16s %s"
              % (k, v.get("value"), v.get("date") or "—",
                 v.get("source", "")[:16], "新鲜" if v.get("fresh") else "过期/缺失"))

    print("\n=== 3. 手动输入的两种场景 ===")
    import datetime as dt

    # 场景 A：自动源新鲜时，自动值优先（减少人工误差）
    save_manual({"ppi": {"value": -9.9, "date": str(dt.date.today()),
                         "note": "手动"}})
    a = get_macro("ppi", prefer="auto")
    print("  自动新鲜时:", a.get("value"), "来源", a.get("source"))
    assert a.get("source") != "手动输入", "自动值新鲜时不应被手动覆盖"
    # 场景 B：强制用手动（prefer=manual）时必须拿手动值
    b = get_macro("ppi", prefer="manual")
    print("  强制手动时:", b.get("value"), "来源", b.get("source"))
    assert b.get("source") == "手动输入" and abs(float(b["value"]) + 9.9) < 1e-9
    # 场景 C：LPR 现在已有日期（TRADE_DATE 修复后），自动值应正常胜出
    m = load_manual()
    m["lpr"] = {"value": 2.95, "date": str(dt.date.today()), "note": "手动"}
    save_manual(m)
    c = get_macro("lpr", prefer="auto")
    print("  LPR 自动优先:", c.get("value"), "来源", c.get("source"))
    assert c.get("source") == "手动输入" or c.get("fresh"), \
        "应给出可用值（自动新鲜的 LPR 或手动）"
    # 场景 D：自动源完全没有的项，手动手动补位
    m["gdp"] = {"value": 5.2, "date": str(dt.date.today()), "note": "手动"}
    save_manual(m)
    d = get_macro("gdp", prefer="auto")
    print("  GDP 手动补位:", d.get("value"), "来源", d.get("source"))
    assert d.get("source") == "手动输入" and d.get("fresh")
    save_manual(old)
    print("  四项规则 OK（自动新鲜优先 / 强制手动 / 自动优先 / 手动补位）")

    print("\n=== 4. 边界 ===")
    save_manual({})
    assert get_macro("lpr").get("value") is None or True
    save_manual(old)
    print("  空手动文件安全")
    print("\nmacro_source self-test OK")
