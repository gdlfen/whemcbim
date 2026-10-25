# -*- coding: utf-8 -*-
"""
"大盘专属「的基本面替代物 —— 指数估值分位、两融余额、北向资金、新增开户数。

为什么大盘需要」替代物"
----------------------
个股有 PE/PB、营收、业绩预告；**指数没有**。但大盘同样有「基本面」：
  · **指数估值分位**：沪深300 的 PE 处于历史什么位置 —— 这是指数的「便宜/贵"
  · **两融余额**：杠杆资金的水位，A股」资金市「的核心指标
  · **北向资金**：外资流向，常被视为聪明的边际资金
  · **新增开户数**：散户入场速度，典型**反向指标**（开户暴增常在阶段顶部）

这四个是 A股大盘最实用的」基本面替代物「，也是官方口径之外的领先/同步指标。

数据源（均已在本机验证可用）
----------------------------
* 指数 PE：``ak.stock_index_pe_lg``（乐咕乐股，259 行日频历史）
* 两融余额：``ak.stock_margin_account_info``（市场整体，3413 行）
* 北向资金：``ak.stock_hsgt_hist_em``（2774 行日频，含当日净买额）
* 新增开户：``ak.stock_account_statistics_em``（101 行**月频**）

重要实现约束
------------
akshare 内含 V8（py_mini_racer），多线程并发会崩进程（已由 core/v8_guard 串行化）。
这四个接口较慢（单次 1~5 秒），所以：
  · 用**进程内 TTL 缓存**（默认 1 小时）避免每次分析都重抓
  · 任一接口失败不影响其他，如实返回」未取得"
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd

from .contracts import Category, DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Factor, safe_float
from .utils import get_logger

log = get_logger()

#: 进程内缓存：key → (取数时间, 值)。这些是日频/月频慢数据，1 小时足够。
_CACHE: dict = {}
_CACHE_TTL = 3600.0
_LOCK = threading.RLock()


def _cached(key: str, fn, ttl: float = _CACHE_TTL):
    """带 TTL 的进程内缓存（线程安全）。"""
    now = time.time()
    with _LOCK:
        got = _CACHE.get(key)
        if got and (now - got[0]) < ttl:
            return got[1]
    try:
        val = fn()
    except Exception as exc:  # noqa: BLE001
        log.debug("取数失败 %s: %s", key, exc)
        val = None
    with _LOCK:
        _CACHE[key] = (now, val)
    return val



def _days_old(date_str) -> float | None:
    """数据日期距今多少天；解析失败返回 None。"""
    try:
        import pandas as _pd

        d = _pd.to_datetime(str(date_str)[:10], errors="coerce")
        if d is None or _pd.isna(d):
            return None
        return float((_pd.Timestamp.now().normalize() - d).days)
    except Exception:  # noqa: BLE001
        return None


#: 各数据源的新鲜度上限（天）。超期则**不生成因子**，只标注"数据过旧"。
MAX_AGE_DAYS = {"pe": 30, "margin": 20, "north": 20, "accounts": 400}


# --------------------------------------------------------------------------- #
# 一、指数估值分位
# --------------------------------------------------------------------------- #
#: 我们的指数 key → 乐咕乐股的指数名
_PE_NAME = {"sh": "上证50", "hs300": "沪深300", "zz500": "中证500",
            "cyb": "创业板指", "sz": "深证成指", "kc50": "科创50"}


def index_valuation(index_key: str = "hs300") -> dict:
    """指数 PE 及其历史分位（乐咕乐股）。"""
    name = _PE_NAME.get(index_key, "沪深300")

    def _get():
        import akshare as ak

        df = ak.stock_index_pe_lg(symbol=name)
        if df is None or not len(df):
            return None
        return df

    df = _cached("pe_%s" % name, _get)
    out: dict = {"name": name, "detail": ""}
    if df is None or not len(df):
        out["detail"] = "未取得 %s 的 PE 历史（乐咕乐股接口）" % name
        return out
    try:
        d = df.copy()
        d["日期"] = pd.to_datetime(d["日期"], errors="coerce")
        d = d.dropna(subset=["日期"]).sort_values("日期")
        # 优先「滚动市盈率」，其次「静态市盈率」（不同指数列名略有差异）
        col = None
        for cand in ("滚动市盈率", "静态市盈率", "等权滚动市盈率", "等权静态市盈率"):
            hit = [c for c in d.columns if cand in str(c)]
            if hit:
                col = hit[0]
                break
        if col is None:
            out["detail"] = "PE 数据缺少可识别的市盈率列：%s" % list(d.columns)[:8]
            return out
        s = pd.to_numeric(d[col], errors="coerce").dropna()
        if not len(s):
            out["detail"] = "PE 序列为空"
            return out
        cur = float(s.iloc[-1])
        pct = float((s <= cur).mean()) * 100
        out.update({
            "pe": round(cur, 2), "percentile": round(pct, 1),
            "median": round(float(s.median()), 2),
            "min": round(float(s.min()), 2), "max": round(float(s.max()), 2),
            "samples": int(len(s)), "column": col,
            "date": str(d["日期"].iloc[-1])[:10],
        })
        try:
            from . import freshness as _FR

            _FR.validate("index_pe", out.get("date"))
        except Exception:  # noqa: BLE001
            pass
        out["detail"] = ("%s PE（%s）%.2f，历史分位 %.1f%%（样本 %d 天，"
                         "区间 %.2f~%.2f，中位数 %.2f），数据日 %s"
                         % (name, col, cur, pct, len(s), out["min"], out["max"],
                            out["median"], out["date"]))
    except Exception as exc:  # noqa: BLE001
        out["detail"] = "PE 解析失败：%s" % str(exc)[:60]
    return out


# --------------------------------------------------------------------------- #
# 二、两融余额
# --------------------------------------------------------------------------- #
def margin_balance() -> dict:
    """市场整体两融余额及其趋势。"""
    def _get():
        import akshare as ak

        df = ak.stock_margin_account_info()
        return df if df is not None and len(df) else None

    df = _cached("margin", _get)
    out: dict = {"detail": ""}
    if df is None:
        out["detail"] = "未取得两融余额数据"
        return out
    try:
        d = df.copy()
        dcol = next((c for c in d.columns if "日期" in str(c)), None)
        fcol = next((c for c in d.columns if "融资余额" in str(c)), None)
        tcol = next((c for c in d.columns if "融资融券余额" in str(c)), None)
        if dcol is None or (fcol is None and tcol is None):
            out["detail"] = "两融数据缺列：%s" % list(d.columns)[:8]
            return out
        d[dcol] = pd.to_datetime(d[dcol], errors="coerce")
        d = d.dropna(subset=[dcol]).sort_values(dcol)
        col = tcol or fcol
        s = pd.to_numeric(d[col], errors="coerce").dropna()
        if not len(s):
            out["detail"] = "两融余额序列为空"
            return out
        cur = float(s.iloc[-1])
        chg5 = ((cur / float(s.iloc[-6]) - 1) * 100) if len(s) > 6 else None
        chg20 = ((cur / float(s.iloc[-21]) - 1) * 100) if len(s) > 21 else None
        pct = float((s <= cur).mean()) * 100
        out.update({
            "balance": round(cur, 2), "change_5d_pct": round(chg5, 2) if chg5 is not None else None,
            "change_20d_pct": round(chg20, 2) if chg20 is not None else None,
            "percentile": round(pct, 1), "samples": int(len(s)),
            "date": str(d[dcol].iloc[-1])[:10], "column": col,
        })
        try:
            from . import freshness as _FR

            _FR.validate("margin", out.get("date"))
        except Exception:  # noqa: BLE001
            pass
        out["detail"] = ("两融余额 %.2f 亿元（%s），近5日 %s%%，近20日 %s%%，"
                         "历史分位 %.1f%%"
                         % (cur, out["date"],
                            ("%+.2f" % chg5) if chg5 is not None else "—",
                            ("%+.2f" % chg20) if chg20 is not None else "—", pct))
    except Exception as exc:  # noqa: BLE001
        out["detail"] = "两融解析失败：%s" % str(exc)[:60]
    return out


# --------------------------------------------------------------------------- #
# 三、北向资金
# --------------------------------------------------------------------------- #
def northbound_flow() -> dict:
    """北向资金净买入及其趋势。"""
    def _get():
        import akshare as ak

        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        return df if df is not None and len(df) else None

    df = _cached("north", _get)
    out: dict = {"detail": ""}
    if df is None:
        out["detail"] = "未取得北向资金数据"
        return out
    try:
        d = df.copy()
        dcol = next((c for c in d.columns if "日期" in str(c)), None)
        vcol = next((c for c in d.columns if "净买额" in str(c) or "资金流入" in str(c)), None)
        if dcol is None or vcol is None:
            out["detail"] = "北向数据缺列：%s" % list(d.columns)[:8]
            return out
        d[dcol] = pd.to_datetime(d[dcol], errors="coerce")
        d = d.dropna(subset=[dcol]).sort_values(dcol)
        s = pd.to_numeric(d[vcol], errors="coerce").dropna()
        if not len(s):
            out["detail"] = "北向净买额为空（该指标 2024-08 起不再逐日披露）"
            return out
        cur = float(s.iloc[-1])
        sum5 = float(s.tail(5).sum())
        sum20 = float(s.tail(20).sum())
        pos_days = int((s.tail(20) > 0).sum())
        out.update({
            "latest": round(cur, 2), "sum_5d": round(sum5, 2),
            "sum_20d": round(sum20, 2), "positive_days_20": pos_days,
            "samples": int(len(s)), "date": str(d[dcol].iloc[-1])[:10],
            "column": vcol,
        })
        try:
            from . import freshness as _FR

            _FR.validate("north", out.get("date"))
        except Exception:  # noqa: BLE001
            pass
        out["detail"] = ("北向资金（%s）当日 %.2f 亿，近5日合计 %.2f 亿，"
                         "近20日合计 %.2f 亿（20日中 %d 天净买入）"
                         % (out["date"], cur, sum5, sum20, pos_days))
    except Exception as exc:  # noqa: BLE001
        out["detail"] = "北向解析失败：%s" % str(exc)[:60]
    return out


# --------------------------------------------------------------------------- #
# 四、新增开户数（月频，反向指标）
# --------------------------------------------------------------------------- #
def new_accounts() -> dict:
    """新增投资者数量（月频）。散户入场速度，典型反向指标。"""
    def _get():
        # 主源：东财 datacenter RPT_STOCK_OPEN_DATA（字段更全：
        # 新增投资者/环比/同比/期末投资者总数，实测与 akshare 同源同止）
        try:
            import requests

            u = ("https://datacenter-web.eastmoney.com/api/data/v1/get?"
                 "reportName=RPT_STOCK_OPEN_DATA&columns=ALL&pageNumber=1"
                 "&pageSize=400&sortColumns=STATISTICS_DATE&sortTypes=-1"
                 "&source=WEB&client=WEB")
            j = requests.get(u, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 Chrome/122",
                "Referer": "https://data.eastmoney.com/"}).json()
            rows = (j.get("result") or {}).get("data") or []
            if rows:
                df = pd.DataFrame(rows)
                df = df.rename(columns={
                    "STATISTICS_DATE": "数据日期",
                    "ADD_INVESTOR": "新增投资者-数量",
                    "ADD_INVESTOR_YOY": "新增投资者-同比",
                    "ADD_INVESTOR_QOQ": "新增投资者-环比",
                    "END_INVESTOR": "期末投资者-总量"})
                return df
        except Exception as exc:  # noqa: BLE001
            log.debug("东财开户数失败，回退 akshare: %s", exc)
        try:
            import akshare as ak

            df = ak.stock_account_statistics_em()
            return df if df is not None and len(df) else None
        except Exception:  # noqa: BLE001
            return None

    df = _cached("accounts", _get)
    out: dict = {"detail": ""}
    if df is None:
        out["detail"] = "未取得新增开户数数据"
        return out
    try:
        d = df.copy()
        dcol = next((c for c in d.columns if "日期" in str(c)), None)
        vcol = next((c for c in d.columns if "新增投资者" in str(c) and "数量" in str(c)), None)
        ycol = next((c for c in d.columns if "新增投资者" in str(c) and "同比" in str(c)), None)
        if dcol is None or vcol is None:
            out["detail"] = "开户数据缺列：%s" % list(d.columns)[:8]
            return out
        d[dcol] = pd.to_datetime(d[dcol], errors="coerce")
        d = d.dropna(subset=[dcol]).sort_values(dcol)
        s = pd.to_numeric(d[vcol], errors="coerce").dropna()
        if not len(s):
            out["detail"] = "开户数序列为空"
            return out
        cur = float(s.iloc[-1])
        pct = float((s <= cur).mean()) * 100
        yoy = None
        if ycol is not None:
            try:
                _y = float(pd.to_numeric(d[ycol], errors="coerce").dropna().iloc[-1])
                yoy = round(_y * 100, 2) if abs(_y) <= 1.5 else round(_y, 2)
            except Exception:  # noqa: BLE001
                yoy = None
        out.update({
            "latest": round(cur, 2), "percentile": round(pct, 1),
            "yoy_pct": yoy, "samples": int(len(s)),
            "date": str(d[dcol].iloc[-1])[:10],
        })
        try:
            from . import freshness as _FR

            _FR.validate("accounts", out.get("date"))
        except Exception:  # noqa: BLE001
            pass
        out["detail"] = ("新增投资者 %.2f（%s，月频），历史分位 %.1f%%%s"
                         % (cur, out["date"], pct,
                            "，同比 %+.2f%%" % yoy if yoy is not None else ""))
    except Exception as exc:  # noqa: BLE001
        out["detail"] = "开户数解析失败：%s" % str(exc)[:60]
    return out


# --------------------------------------------------------------------------- #
# 五、因子
# --------------------------------------------------------------------------- #
def fundamental_factors(index_key: str = "hs300",
                        pe: dict | None = None,
                        margin: dict | None = None,
                        north: dict | None = None,
                        accounts: dict | None = None) -> list[Factor]:
    """把四个替代物转成因子（用于指数预测）。"""
    out: list[Factor] = []
    pe = pe if pe is not None else index_valuation(index_key)
    margin = margin if margin is not None else margin_balance()
    north = north if north is not None else northbound_flow()
    accounts = accounts if accounts is not None else new_accounts()

    # ---- 新鲜度校验 ----
    # 实测 akshare 的开户数只更新到 2023-08（三年旧），若当"当前"用会误导。
    # 规则：超过 MAX_AGE_DAYS 的数据一律不生成因子，只保留说明。
    stale: dict = {}
    for key, d in (("pe", pe), ("margin", margin), ("north", north),
                   ("accounts", accounts)):
        age = _days_old(d.get("date")) if d.get("detail") else None
        limit = MAX_AGE_DAYS.get(key, 30)
        if age is not None and age > limit:
            stale[key] = (age, limit)
            d["stale"] = True
            d["detail"] = ("%s（**数据过旧：距今 %.0f 天，超过 %d 天上限，"
                            "不作为当前信号**）"
                            % (d.get("detail") or "", age, limit))
    if stale.get("pe"):
        pe = {**pe, "percentile": None}
    if stale.get("margin"):
        margin = {**margin, "change_20d_pct": None}
    if stale.get("north"):
        north = {**north, "sum_20d": None}
    if stale.get("accounts"):
        accounts = {**accounts, "percentile": None}

    # ---- 1) 指数估值分位（长周期最有效）---- #
    pct = pe.get("percentile")
    if pct is not None:
        if pct <= 20:
            out.append(Factor(
                "mf_valuation", "指数估值处于低位（%.0f%% 分位）" % pct,
                Category.MACRO, DIR_BULL, 0.62, 0.75,
                value="PE %.2f（%.0f%% 分位）" % (pe.get("pe") or 0, pct),
                detail="%s。估值是长周期最可靠的锚——**PE 分位比「故事」更可靠**，"
                       "低位区间的下跌空间有限、向上弹性更大。"
                       % (pe.get("detail") or ""),
                theory="估值分位/均值回归", horizon=("m1", "q1", "h1", "y1")))
        elif pct >= 80:
            out.append(Factor(
                "mf_valuation", "指数估值处于高位（%.0f%% 分位）" % pct,
                Category.MACRO, DIR_BEAR, 0.62, 0.75,
                value="PE %.2f（%.0f%% 分位）" % (pe.get("pe") or 0, pct),
                detail="%s。估值处于历史高位，继续上行需要盈利兑现或流动性进一步宽松，"
                       "否则均值回归压力会逐步显现。" % (pe.get("detail") or ""),
                theory="估值分位/均值回归", horizon=("m1", "q1", "h1", "y1")))
        else:
            out.append(Factor(
                "mf_valuation", "指数估值中性（%.0f%% 分位）" % pct,
                Category.MACRO, DIR_NEUTRAL, 0.2, 0.7,
                value="PE %.2f（%.0f%% 分位）" % (pe.get("pe") or 0, pct),
                detail=pe.get("detail") or "",
                theory="估值分位", horizon=("q1", "h1", "y1")))

    # ---- 2) 两融余额（杠杆资金水位，A股「资金市」核心）---- #
    chg20 = margin.get("change_20d_pct")
    m_pct = margin.get("percentile")
    if chg20 is not None:
        if chg20 >= 3.0:
            out.append(Factor(
                "mf_margin", "两融余额加速上升（近20日 %+.1f%%）" % chg20,
                Category.CAPITAL, DIR_BULL, min(chg20 / 15.0 + 0.25, 0.75), 0.7,
                value="%+.2f%%（余额 %.0f 亿）" % (chg20, margin.get("balance") or 0),
                detail="%s。两融余额是**杠杆资金的水位**——A股是「资金市」，"
                       "水涨船高：两融扩张通常对应风险偏好上升、行情延续。"
                       % (margin.get("detail") or ""),
                theory="资金市/杠杆资金", horizon=("d20", "m1", "q1")))
        elif chg20 <= -3.0:
            out.append(Factor(
                "mf_margin", "两融余额快速收缩（近20日 %+.1f%%）" % chg20,
                Category.CAPITAL, DIR_BEAR, min(abs(chg20) / 15.0 + 0.25, 0.75), 0.7,
                value="%+.2f%%（余额 %.0f 亿）" % (chg20, margin.get("balance") or 0),
                detail="%s。两融快速收缩说明杠杆资金在撤离，"
                       "往往伴随下跌或加剧下跌（被动平仓压力）。"
                       % (margin.get("detail") or ""),
                theory="资金市/杠杆资金", horizon=("d20", "m1", "q1")))

    # ---- 3) 北向资金（边际聪明钱）---- #
    s20 = north.get("sum_20d")
    if s20 is not None:
        if s20 > 50:
            out.append(Factor(
                "mf_northbound", "北向资金持续净买入（近20日 %+.1f 亿）" % s20,
                Category.CAPITAL, DIR_BULL, min(abs(s20) / 400.0 + 0.25, 0.72), 0.66,
                value="%+.1f 亿" % s20,
                detail="%s。北向资金被视为**边际聪明钱**，持续净买入通常与"
                       "估值修复或外资增配同步。" % (north.get("detail") or ""),
                theory="外资流向", horizon=("d10", "d20", "m1")))
        elif s20 < -50:
            out.append(Factor(
                "mf_northbound", "北向资金持续净卖出（近20日 %+.1f 亿）" % s20,
                Category.CAPITAL, DIR_BEAR, min(abs(s20) / 400.0 + 0.25, 0.72), 0.66,
                value="%+.1f 亿" % s20,
                detail="%s。北向持续净卖出多为外资减配或全球风险偏好回落。"
                       % (north.get("detail") or ""),
                theory="外资流向", horizon=("d10", "d20", "m1")))

    # ---- 4) 新增开户数（散户情绪，反向指标）---- #
    # ---- 新增开户数：**已停更，不再生成因子** ----
    # 中登自 2023-08 起停止公开发布月度新增投资者数（akshare 与东财
    # RPT_STOCK_OPEN_DATA 均止于此，上交所/深交所/中登官网均无法自动获取）。
    # 散户入场意愿改用 core/retail_heat.py 的代理指标
    # （成交额水位 / 振幅热度 / 两融变化，均为日频新鲜数据）。
    a_pct = None
    a_pct = accounts.get("percentile")
    if a_pct is not None:
        if a_pct >= 85:
            out.append(Factor(
                "mf_new_accounts", "新增开户数激增（%.0f%% 分位）" % a_pct,
                Category.SENTIMENT, DIR_BEAR, 0.58, 0.6,
                value="%.2f（%.0f%% 分位）" % (accounts.get("latest") or 0, a_pct),
                detail="%s。**散户开户暴增往往是阶段顶部的信号**"
                       "（反向指标）——牛市末期开户数通常创纪录。"
                       % (accounts.get("detail") or ""),
                theory="散户情绪/反向指标", horizon=("d20", "m1", "q1")))
        elif a_pct <= 15:
            out.append(Factor(
                "mf_new_accounts", "新增开户数低迷（%.0f%% 分位）" % a_pct,
                Category.SENTIMENT, DIR_BULL, 0.55, 0.6,
                value="%.2f（%.0f%% 分位）" % (accounts.get("latest") or 0, a_pct),
                detail="%s。开户低迷说明散户情绪冰点，"
                       "历史上这往往对应底部区域（反向指标）。"
                       % (accounts.get("detail") or ""),
                theory="散户情绪/反向指标", horizon=("d20", "m1", "q1")))
    return out


# --------------------------------------------------------------------------- #
# 六、一站式
# --------------------------------------------------------------------------- #
def analyze(index_key: str = "hs300") -> dict:
    """取齐四个替代物并生成因子。"""
    pe = index_valuation(index_key)
    margin = margin_balance()
    north = northbound_flow()
    accounts = new_accounts()
    fs = fundamental_factors(index_key, pe, margin, north, accounts)
    bits = [d.get("detail") for d in (pe, margin, north, accounts) if d.get("detail")]
    return {"pe": pe, "margin": margin, "north": north, "accounts": accounts,
            "factors": fs,
            # stale 是 fundamental_factors 的局部变量，这里从各数据源的
            # d[stale] 标记反推（跨函数访问局部变量是上一版的 bug）
            "stale": [k for k, d in (("估值", pe), ("两融", margin),
                                      ("北向", north), ("开户数", accounts))
                       if isinstance(d, dict) and d.get("stale")],
            "detail": "；".join(bits)}


if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")

    print("=== 1. 指数估值分位 ===")
    pe = index_valuation("hs300")
    print(" ", pe.get("detail"))
    print("   PE=%s 分位=%s 样本=%s" % (pe.get("pe"), pe.get("percentile"),
                                    pe.get("samples")))

    print("\n=== 2. 两融余额 ===")
    mg = margin_balance()
    print(" ", mg.get("detail"))

    print("\n=== 3. 北向资金 ===")
    nt = northbound_flow()
    print(" ", nt.get("detail"))

    print("\n=== 4. 新增开户数 ===")
    ac = new_accounts()
    print(" ", ac.get("detail"))

    print("\n=== 5. 因子 ===")
    fs = fundamental_factors("hs300", pe, mg, nt, ac)
    print("  生成 %d 条因子" % len(fs))
    for f in fs:
        print("     %-34s %s" % (f.name[:32], {1: "多", -1: "空", 0: "中"}[f.direction]))
    for f in fs:
        assert f.detail and f.theory
        assert 0 <= f.strength <= 1 and 0 <= f.reliability <= 1
        assert "None" not in str(f.value)

    print("\n=== 6. 边界 ===")
    e = fundamental_factors("hs300", {}, {}, {}, {})
    assert e == [], "全空输入不应产生因子"
    assert index_valuation("nonexistent").get("detail")
    print("  空输入安全")

    print("\n=== 7. 缓存生效 ===")
    t0 = time.time()
    margin_balance()
    dt = time.time() - t0
    print("  第二次取两融耗时 %.3f 秒（应接近 0，走缓存）" % dt)
    assert dt < 0.5, "缓存应生效"
    print("\nmarket_fundamental self-test OK")
