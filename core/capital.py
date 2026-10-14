# -*- coding: utf-8 -*-
"""
资金面与主力动向模块（Capital）。

职责
----
把**主力资金流向、大单结构、龙虎榜、北向持股、股东户数、前十大流通股东、两融余额**
压缩成 :class:`core.contracts.Factor`，回答"钱在往哪走、谁在控盘、主力在吃货还是吐货"。

产出因子 id
-----------
``capital_main_net``（主力净流入 5/10/20 日加权）、``capital_super_big``（超大单/大单）、
``capital_main_ratio``（主力净占比）、``capital_trend_consistency``（连续净流入天数）、
``capital_divergence``（价量资金背离）、``lhb_institution``（机构席位净买）、
``northbound_change``（北向增减持）、``holder_concentration``（股东户数→筹码集中度）、
``top10_float_change``（前十大流通股东增减持）、``fund_holding_change``（基金持仓，取不到则跳过）、
``margin_balance_trend``（两融余额趋势）、``control_degree``（控盘程度）、
``main_force_behavior``（主力行为：吃货/洗盘/吐货/出货）。

数据源与实测结论
----------------
* **东财资金流**（``push2his/push2 .../fflow/daykline/get``）在本机**整体不可用**
  （连续 4 次 ``RemoteDisconnected``，含加 ``ut``/``Referer``/换主机/akshare 封装等 8 种变体），
  因此 :func:`fetch_fund_flow` 把它放在降级链首位（其它网络环境通常可用），
  并以**新浪资金流**为主力兜底（实测可用）。
* 新浪资金流 ``MoneyFlow.ssl_qsfx_lscjfb``（需 ``Referer: finance.sina.com.cn``）字段实测：
  ``opendate, trade(收盘), changeratio(涨跌幅小数), turnover, netamount(净流入元),
  ratioamount(净流入率), r0/r1/r2/r3(超大单/大单/中单/小单成交额),
  r0_net/r1_net/r2_net/r3_net(各级净额)``；实测 ``r0_net+r1_net+r2_net+r3_net == netamount``。
  本模块定义 **主力净流入 = r0_net + r1_net**（超大单 + 大单，与东财"主力"口径一致）。
* 东财 ``fflow/daykline`` 的 ``data.klines`` 字段映射（用户给定 + 通行口径，代码中保留解析）：
  ``0=日期, 1=主力净流入, 2=小单净流入, 3=中单净流入, 4=大单净流入, 5=超大单净流入,
  6=主力净占比, 7=小单净占比, 8=中单净占比, 9=大单净占比, 10=超大单净占比, 11=收盘价,
  12=涨跌幅``（本机无法实测，故解析时做长度与数值健壮性校验，并在注释中标注"未实测"）。
* 股东户数：akshare ``stock_zh_a_gdhs_detail_em``（实测 63 行、**升序**）→ 东财
  ``RPT_HOLDERNUMLATEST``（实测仅最新 1 行）。
* 两融：``RPTA_WEB_RZRQ_GGMX``（实测可用，字段 ``RZYE/RZMRE/RQYL/RQYE/RZRQYE/RZYEZB/RZJME``）。
* 龙虎榜：``RPT_DAILYBILLBOARD_DETAILSNEW`` + 机构席位 ``RPT_ORGANIZATION_TRADE_DETAILS``。
* 北向：``RPT_MUTUAL_HOLDSTOCKNORTH_STA``（**不能带 sortColumns=HOLD_DATE，会报表不存在**）。
* 十大流通股东：``RPT_F10_EH_FREEHOLDERS``。

所有联网函数优雅降级：失败返回空 ``DataFrame`` / ``{}`` 并记录中文说明，绝不抛异常。
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd

from . import config
from .contracts import (Category, DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Factor,
                        clamp, now_cn, safe_float)
from .utils import (cache_key, cached, get_logger, http_get, normalize_symbol,
                    retry_call)

log = get_logger("stockai.capital")

__all__ = [
    "fetch_fund_flow", "fetch_minute_flow", "fetch_holders", "fetch_margin",
    "fetch_lhb", "fetch_northbound", "fetch_top10_float_holders",
    "capital_factors", "control_degree", "main_force_profile",
    "fetch_snapshot", "last_notes", "FLOW_COLUMNS",
]

#: 规范资金流列（单位：元；``date`` 为 Timestamp，升序）
FLOW_COLUMNS = ["date", "main_net", "small_net", "mid_net", "big_net", "super_net",
                "main_pct", "big_pct", "super_pct", "close", "pct_chg",
                "main_net_5d", "main_net_10d", "main_net_20d"]

_EM_FFLOW_DAY = ("https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
                 "?secid=%s&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,"
                 "f58,f59,f60,f61,f62,f63,f64,f65&klt=101&lmt=%d")
_EM_FFLOW_MIN = ("https://push2his.eastmoney.com/api/qt/stock/fflow/kline/get"
                 "?secid=%s&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,"
                 "f58,f59,f60,f61,f62,f63,f64,f65&klt=1&lmt=0")
#: 东财资金流备用主机（本机 push2his/push2 均间歇性拒连，其它网络环境可命中）
_EM_FFLOW_HOSTS = ("https://push2his.eastmoney.com", "https://push2.eastmoney.com",
                   "https://push2delay.eastmoney.com")
_SINA_FLOW = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
              "MoneyFlow.ssl_qsfx_lscjfb?page=1&num=%d&sort=opendate&asc=0&daima=%s")
_EM_DATACENTER = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_TENCENT_URL = "https://qt.gtimg.cn/q=%s"

#: 腾讯快照实测索引（详见 core/fundamental.py 的 TENCENT_FIELD 注释）
_TX_IDX = {"name": 1, "price": 3, "prev_close": 4, "volume": 6, "pct_chg": 32,
           "amount_wan": 37, "turnover": 38, "pe_ttm": 39, "amplitude": 43,
           "float_market_cap_yi": 44, "market_cap_yi": 45, "pb": 46,
           "volume_ratio": 49, "total_shares": 72, "float_shares": 73}

#: 资金类因子适用周期（短线为主）
FLOW_HORIZON = ("d1", "d5", "d10", "d20", "m1")

#: 主力行为判定阈值（注释见 :func:`main_force_profile`）
FORCE_TH = {"strong_net": 0.15, "mild_up": 8.0, "resist_drop": -3.0, "breakdown": -8.0}

_LAST_NOTES: list = []


def last_notes() -> list:
    """最近一次取数的降级说明（中文）。"""
    return list(_LAST_NOTES)


def _note(msg: str) -> None:
    if msg and msg not in _LAST_NOTES:
        _LAST_NOTES.append(msg)
    log.debug(msg)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _num(x, default=float("nan")) -> float:
    return safe_float(x, default)


def _ensure_frame(df):
    """列名容错（大写/中文 → 小写英文），非 DataFrame 返回 None。"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    ren = {"日期": "date", "时间": "date", "Date": "date", "DATE": "date",
           "收盘": "close", "开盘": "open", "最高": "high", "最低": "low",
           "成交量": "volume", "成交额": "amount", "换手率": "turnover",
           "涨跌幅": "pct_chg", "Close": "close", "Volume": "volume"}
    return df.rename(columns={k: v for k, v in ren.items() if k in df.columns}).copy()


def _code(symbol) -> str:
    try:
        return normalize_symbol(symbol)["code"]
    except Exception:  # noqa: BLE001
        return re.sub(r"\D", "", str(symbol or ""))[:6]


def _secid(symbol) -> str:
    try:
        return normalize_symbol(symbol)["secid"]
    except Exception:  # noqa: BLE001
        return "1.600519"


def _prefix(symbol) -> str:
    try:
        return normalize_symbol(symbol)["display"]
    except Exception:  # noqa: BLE001
        return "sh600519"


def _secucode(symbol) -> str:
    try:
        info = normalize_symbol(symbol)
        return info["code"] + ("." + info["prefix"].upper() if info["is_cn"] else "")
    except Exception:  # noqa: BLE001
        return "600519.SH"


def _em_query(report: str, filter_expr: str, page_size: int = 60, sort: str = "",
              sort_type: str = "-1", attempts: int = 3) -> list:
    """东财 datacenter-web 通用查询，返回 ``result.data`` 列表（失败返回 []）。"""
    url = (_EM_DATACENTER + "?reportName=%s&columns=ALL&filter=%s&pageSize=%d"
           % (report, filter_expr, page_size))
    if sort:
        url += "&sortColumns=%s&sortTypes=%s" % (sort, sort_type)
    payload = retry_call(lambda: http_get(url, json_body=True), attempts=attempts,
                         delay=0.7, default=None, label=report)
    if not payload:
        _note("%s 请求失败（已降级为空）" % report)
        return []
    rows = ((payload.get("result") or {}).get("data")) or []
    if not rows:
        _note("%s 返回 0 行：%s" % (report, payload.get("message")))
    return rows


def _neutral(fid: str, name: str, detail: str, reliability: float = 0.2, value=None,
             theory: str = "资金流向理论", horizon=FLOW_HORIZON) -> Factor:
    """数据缺失时的中性因子（保证因子 id 集合稳定）。"""
    return Factor(fid, name, Category.CAPITAL, DIR_NEUTRAL, 0.0,
                  clamp(reliability, 0.05, 0.5), value=value, detail=detail,
                  theory=theory, horizon=horizon)


# --------------------------------------------------------------------------- #
# 1. 资金流
# --------------------------------------------------------------------------- #
def _parse_em_klines(klines: list) -> pd.DataFrame:
    """解析东财资金流 ``data.klines``（每行逗号分隔）。

    字段映射（**本机未实测**，因 push2his 拒连；按东财通行口径解析并做健壮性校验）::

        [0] 日期        [1] 主力净流入    [2] 小单净流入   [3] 中单净流入
        [4] 大单净流入  [5] 超大单净流入  [6] 主力净占比   [7] 小单净占比
        [8] 中单净占比  [9] 大单净占比    [10] 超大单净占比 [11] 收盘价  [12] 涨跌幅
    """
    rows = []
    for line in klines or []:
        c = str(line).split(",")
        if len(c) < 11:
            continue
        rows.append({
            "date": pd.Timestamp(str(c[0])),
            "main_net": _num(c[1]), "small_net": _num(c[2]), "mid_net": _num(c[3]),
            "big_net": _num(c[4]), "super_net": _num(c[5]),
            "main_pct": _num(c[6]), "big_pct": _num(c[9]) if len(c) > 9 else np.nan,
            "super_pct": _num(c[10]) if len(c) > 10 else np.nan,
            "close": _num(c[11]) if len(c) > 11 else np.nan,
            "pct_chg": _num(c[12]) if len(c) > 12 else np.nan,
        })
    return pd.DataFrame(rows)


def _sina_flow_frame(symbol, days: int) -> pd.DataFrame:
    """新浪资金流兜底通道（实测可用）。

    字段：``opendate, trade, changeratio(小数), turnover, netamount, ratioamount,
    r0/r1/r2/r3(超大/大/中/小单成交额), r0_net..r3_net(对应净额)``。
    主力 = 超大单 + 大单；占比用「净额 ÷ 当日总成交额(r0+r1+r2+r3)」计算（与新浪
    ``ratioamount`` 口径一致，实测 2026-10-09：主力 6.27 亿 / 44.0 亿 ≈ 14.2%）。
    """
    txt = http_get(_SINA_FLOW % (max(10, int(days)), _prefix(symbol)),
                   headers={"Referer": "https://finance.sina.com.cn"}, encoding="gbk")
    data = json.loads(txt) if txt else []
    rows = []
    for r in data or []:
        r0, r1 = _num(r.get("r0_net")), _num(r.get("r1_net"))
        amt = sum(_num(r.get(k)) for k in ("r0", "r1", "r2", "r3"))
        main = (0.0 if math.isnan(r0) else r0) + (0.0 if math.isnan(r1) else r1)
        rows.append({
            "date": pd.Timestamp(str(r.get("opendate"))),
            "main_net": main,
            "small_net": _num(r.get("r3_net")), "mid_net": _num(r.get("r2_net")),
            "big_net": r1, "super_net": r0,
            "main_pct": (main / amt * 100.0) if amt > 0 else np.nan,
            "big_pct": (r1 / amt * 100.0) if (amt > 0 and not math.isnan(r1)) else np.nan,
            "super_pct": (r0 / amt * 100.0) if (amt > 0 and not math.isnan(r0)) else np.nan,
            "close": _num(r.get("trade")),
            "pct_chg": _num(r.get("changeratio")) * 100.0,
        })
    return pd.DataFrame(rows)


def _finalize_flow(df: pd.DataFrame) -> pd.DataFrame:
    """补齐列、排序、派生 5/10/20 日滚动主力净流入。"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.copy()
    for c in FLOW_COLUMNS[:11]:
        if c not in df.columns:
            df[c] = np.nan
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    if df.empty:
        return pd.DataFrame()
    for c in FLOW_COLUMNS[1:11]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for w in (5, 10, 20):
        df["main_net_%dd" % w] = df["main_net"].rolling(w, min_periods=1).sum()
    return df[FLOW_COLUMNS].reset_index(drop=True)


def _producer_fund_flow(symbol, days: int) -> pd.DataFrame:
    """资金流取数（不含缓存）：东财多通道 → akshare → 新浪 → 空表。"""
    secid = _secid(symbol)
    # ---- 通道 1~3：东财（本机不可用，其它环境通常可用） ------------------ #
    for host in _EM_FFLOW_HOSTS:
        url = (_EM_FFLOW_DAY % (secid, max(30, int(days)))).replace(
            "https://push2his.eastmoney.com", host)
        for extra in ("", "&ut=b2884a393a59ad64002292a3e90d46a5"):
            payload = retry_call(lambda: http_get(url + extra, json_body=True),
                                 attempts=2, delay=0.5, default=None,
                                 label="fflow_daykline")
            klines = ((payload or {}).get("data") or {}).get("klines") or []
            if klines:
                out = _finalize_flow(_parse_em_klines(klines))
                if not out.empty:
                    _note("资金流来源：东财 fflow/daykline（%s，%d 日）" % (host, len(out)))
                    return out
    _note("东财资金流接口（push2his/push2/push2delay）本机全部拒连："
          "RemoteDisconnected，已降级到 akshare/新浪")
    # ---- 通道 4：akshare 封装 ------------------------------------------- #
    try:
        import akshare as ak
        market = "sh" if _prefix(symbol).startswith("sh") else "sz"
        raw = retry_call(lambda: ak.stock_individual_fund_flow(stock=_code(symbol),
                                                              market=market),
                         attempts=2, delay=0.8, default=None, label="ak_individual_flow")
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            ren = {"日期": "date", "主力净流入-净额": "main_net", "小单净流入-净额": "small_net",
                   "中单净流入-净额": "mid_net", "大单净流入-净额": "big_net",
                   "超大单净流入-净额": "super_net", "主力净流入-净占比": "main_pct",
                   "大单净流入-净占比": "big_pct", "超大单净流入-净占比": "super_pct",
                   "收盘价": "close", "涨跌幅": "pct_chg"}
            sub = raw.rename(columns={k: v for k, v in ren.items() if k in raw.columns})
            out = _finalize_flow(sub)
            if not out.empty:
                _note("资金流来源：akshare.stock_individual_fund_flow（%d 日）" % len(out))
                return out
    except Exception as exc:  # noqa: BLE001
        _note("akshare 个股资金流失败：%r" % exc)
    # ---- 通道 5：新浪（本机唯一可用通道） ------------------------------- #
    try:
        out = _finalize_flow(_sina_flow_frame(symbol, days))
        if not out.empty:
            _note("资金流来源：新浪 MoneyFlow.ssl_qsfx_lscjfb（%d 日，主力=超大单+大单，"
                  "与东财口径一致）" % len(out))
            return out
    except Exception as exc:  # noqa: BLE001
        _note("新浪资金流失败：%r" % exc)
    _note("资金流所有通道均失败，返回空表（资金面因子退化为中性）")
    return pd.DataFrame()


def fetch_fund_flow(symbol, days=120, force=False) -> pd.DataFrame:
    """日频资金流（单位：元，正数=净流入）。

    Returns
    -------
    pandas.DataFrame
        列见 :data:`FLOW_COLUMNS`：``date, main_net, small_net, mid_net, big_net,
        super_net, main_pct, big_pct, super_pct, close, pct_chg, main_net_5d,
        main_net_10d, main_net_20d``（升序，滚动和为派生列）。
        全部通道失败时返回空 ``DataFrame``。
    """
    code = _code(symbol)
    key = cache_key("flow", code, int(days))

    def _p():
        return _producer_fund_flow(symbol, days)

    try:
        if force:
            return _finalize_flow(_p())
        out = cached(key, config.RUNTIME["cache_ttl_seconds"], _p)
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        _note("fetch_fund_flow 异常降级：%r" % exc)
        return pd.DataFrame()


def fetch_minute_flow(symbol, force=False) -> dict:
    """当日分时资金流汇总。

    Returns
    -------
    dict
        ``{"main_net","super_net","big_net","mid_net","small_net","main_pct","big_pct",
        "super_pct","close","pct_chg","points","source"}``；失败返回 ``{}``。

    备注
    ----
    东财分时资金流（``fflow/kline/get?klt=1``）在本机与日频同样被拒连，
    新浪无对应分钟级接口，故本机返回 ``{}``（已优雅降级，不影响其它因子）。
    """
    code = _code(symbol)
    key = cache_key("flowmin", code)

    def _p() -> dict:
        secid = _secid(symbol)
        for host in _EM_FFLOW_HOSTS:
            url = _EM_FFLOW_MIN.replace("https://push2his.eastmoney.com", host) % secid
            payload = retry_call(lambda: http_get(url, json_body=True), attempts=2,
                                 delay=0.5, default=None, label="fflow_min")
            klines = ((payload or {}).get("data") or {}).get("klines") or []
            df = _parse_em_klines(klines)
            if df.empty:
                continue
            amt = _num(df["main_net"].abs().sum())
            out = {
                "main_net": float(df["main_net"].sum()), "super_net": float(df["super_net"].sum()),
                "big_net": float(df["big_net"].sum()), "mid_net": float(df["mid_net"].sum()),
                "small_net": float(df["small_net"].sum()),
                "main_pct": float(df["main_pct"].iloc[-1]) if df["main_pct"].notna().any()
                else np.nan,
                "big_pct": float(df["big_pct"].iloc[-1]) if df["big_pct"].notna().any()
                else np.nan,
                "super_pct": float(df["super_pct"].iloc[-1]) if df["super_pct"].notna().any()
                else np.nan,
                "close": float(df["close"].iloc[-1]) if df["close"].notna().any() else np.nan,
                "pct_chg": float(df["pct_chg"].iloc[-1]) if df["pct_chg"].notna().any()
                else np.nan,
                "points": int(len(df)), "amount_abs": amt, "source": host,
            }
            _note("分时资金流来源：%s（%d 个点）" % (host, len(df)))
            return out
        _note("分时资金流：东财 klt=1 接口本机拒连，无备选源，返回空 dict")
        return {}

    try:
        if force:
            return _p() or {}
        out = cached(key, 300, _p)      # 分时数据 5 分钟缓存
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001
        _note("fetch_minute_flow 异常降级：%r" % exc)
        return {}


# --------------------------------------------------------------------------- #
# 2. 股东户数
# --------------------------------------------------------------------------- #
def _producer_holders(symbol) -> pd.DataFrame:
    """股东户数（akshare 明细 → 东财最新快照 → 空表）。"""
    code = _code(symbol)
    # ---- akshare 明细（实测 63 行，升序） -------------------------------- #
    try:
        import akshare as ak
        raw = retry_call(lambda: ak.stock_zh_a_gdhs_detail_em(symbol=code), attempts=2,
                         delay=0.8, default=None, label="gdhs_detail_em")
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            df = pd.DataFrame({
                "截止日": pd.to_datetime(raw.get("股东户数统计截止日"), errors="coerce"),
                "股东户数": pd.to_numeric(raw.get("股东户数-本次"), errors="coerce"),
                "户均持股数": pd.to_numeric(raw.get("户均持股数量"), errors="coerce"),
                "户均持股金额": pd.to_numeric(raw.get("户均持股市值"), errors="coerce"),
                "股东户数环比": pd.to_numeric(raw.get("股东户数-增减比例"), errors="coerce"),
                "总市值": pd.to_numeric(raw.get("总市值"), errors="coerce"),
            }).dropna(subset=["截止日"]).sort_values("截止日", ascending=False)
            df = df.reset_index(drop=True)
            # 同比：与约 4 期前（同月份）比较
            yoy = []
            for i, r in df.iterrows():
                target = r["截止日"] - pd.DateOffset(years=1)
                cand = df.iloc[i:][abs((df.iloc[i:]["截止日"] - target).dt.days) <= 45]
                base = _num(cand.iloc[0]["股东户数"]) if len(cand) else float("nan")
                cur = _num(r["股东户数"])
                yoy.append((cur / base - 1) * 100 if base and base > 0 and not math.isnan(base)
                           else float("nan"))
            df["股东户数同比"] = yoy
            _note("股东户数来源：akshare.stock_zh_a_gdhs_detail_em（%d 期）" % len(df))
            return df[["截止日", "股东户数", "户均持股数", "户均持股金额",
                       "股东户数环比", "股东户数同比", "总市值"]]
    except Exception as exc:  # noqa: BLE001
        _note("akshare 股东户数明细失败：%r" % exc)
    # ---- 东财最新快照（实测仅 1 行） ------------------------------------- #
    rows = _em_query("RPT_HOLDERNUMLATEST", '(SECURITY_CODE="%s")' % code, page_size=5,
                     sort="END_DATE")
    if rows:
        r = rows[0]
        df = pd.DataFrame([{
            "截止日": pd.to_datetime(str(r.get("END_DATE"))[:10], errors="coerce"),
            "股东户数": _num(r.get("HOLDER_NUM")),
            "户均持股数": _num(r.get("AVG_HOLD_NUM")),
            "户均持股金额": _num(r.get("AVG_MARKET_CAP")),
            "股东户数环比": _num(r.get("HOLDER_NUM_RATIO")),
            "股东户数同比": np.nan,
            "总市值": _num(r.get("TOTAL_MARKET_CAP")),
        }])
        _note("股东户数降级来源：东财 RPT_HOLDERNUMLATEST（仅最新 1 期，无同比）")
        return df
    _note("股东户数全部来源失败，返回空表")
    return pd.DataFrame()


def fetch_holders(symbol, force=False) -> pd.DataFrame:
    """股东户数变化（筹码集中度的代理指标）。

    Returns
    -------
    pandas.DataFrame
        ``[截止日, 股东户数, 户均持股数, 户均持股金额, 股东户数环比, 股东户数同比]``
        （按截止日降序）。拿不到返回空表（**不影响其它因子**）。
    """
    code = _code(symbol)
    try:
        if force:
            return _producer_holders(symbol)
        out = cached(cache_key("holder", code), config.RUNTIME["cache_ttl_daily"],
                     lambda: _producer_holders(symbol))
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        _note("fetch_holders 异常降级：%r" % exc)
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# 3. 两融
# --------------------------------------------------------------------------- #
def _producer_margin(symbol, days: int) -> pd.DataFrame:
    """两融明细（东财 ``RPTA_WEB_RZRQ_GGMX``，实测可用）。

    真实字段：``DATE, RZYE(融资余额), RZMRE(融资买入额), RQYL(融券余量),
    RQYE(融券余额), RZRQYE(融资融券余额), RZYEZB(融资余额占流通市值比%),
    RZJME(融资净买入), RZCHE(融资偿还额), SPJ(收盘价), ZDF(涨跌幅%)``。
    """
    rows = _em_query("RPTA_WEB_RZRQ_GGMX", '(SCODE="%s")' % _code(symbol),
                     page_size=max(10, int(days)), sort="DATE")
    if not rows:
        return pd.DataFrame()
    recs = []
    for r in rows:
        recs.append({
            "date": pd.to_datetime(str(r.get("DATE"))[:10], errors="coerce"),
            "融资余额": _num(r.get("RZYE")), "融资买入额": _num(r.get("RZMRE")),
            "融券余量": _num(r.get("RQYL")), "融券余额": _num(r.get("RQYE")),
            "融资融券余额": _num(r.get("RZRQYE")),
            "融资余额占流通市值比": _num(r.get("RZYEZB")),
            "融资净买入": _num(r.get("RZJME")), "融资偿还额": _num(r.get("RZCHE")),
            "收盘价": _num(r.get("SPJ")), "涨跌幅": _num(r.get("ZDF")),
        })
    df = pd.DataFrame(recs).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    _note("两融来源：东财 RPTA_WEB_RZRQ_GGMX（%d 个交易日）" % len(df))
    return df


def fetch_margin(symbol, days=60, force=False) -> pd.DataFrame:
    """融资融券明细（升序）。列见 :func:`_producer_margin`；失败返回空表。"""
    code = _code(symbol)
    key = cache_key("margin", code, int(days))
    try:
        if force:
            return _producer_margin(symbol, days)
        out = cached(key, config.RUNTIME["cache_ttl_daily"],
                     lambda: _producer_margin(symbol, days))
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        _note("fetch_margin 异常降级：%r" % exc)
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# 4. 龙虎榜
# --------------------------------------------------------------------------- #
def _producer_lhb(symbol) -> pd.DataFrame:
    """龙虎榜明细 + 机构席位净买（两个报表合并）。

    实测字段（``RPT_DAILYBILLBOARD_DETAILSNEW``）：``TRADE_DATE, EXPLAIN(解读),
    EXPLANATION(上榜原因), BILLBOARD_DEAL_AMT(龙虎榜成交额), DEAL_AMOUNT_RATIO(占总成交比%),
    BILLBOARD_NET_AMT(净买入额), BILLBOARD_BUY_AMT, BILLBOARD_SELL_AMT``；
    机构席位（``RPT_ORGANIZATION_TRADE_DETAILS``）：``TRADE_DATE, BUY_AMT, SELL_AMT,
    NET_BUY_AMT, BUY_TIMES, SELL_TIMES, RATIO``。
    注：600519 历史仅上榜 2 次（2013 年），属正常现象。
    """
    rows = _em_query("RPT_DAILYBILLBOARD_DETAILSNEW",
                     '(SECURITY_CODE="%s")' % _code(symbol), page_size=30,
                     sort="TRADE_DATE")
    org = _em_query("RPT_ORGANIZATION_TRADE_DETAILS",
                    '(SECURITY_CODE="%s")' % _code(symbol), page_size=30, sort="TRADE_DATE")
    org_map = {}
    for r in org:
        d = str(r.get("TRADE_DATE"))[:10]
        org_map[d] = _num(r.get("NET_BUY_AMT"))
    if not rows and not org_map:
        return pd.DataFrame()
    recs = []
    for r in rows:
        d = str(r.get("TRADE_DATE"))[:10]
        recs.append({
            "date": pd.to_datetime(d, errors="coerce"),
            "上榜原因": r.get("EXPLANATION") or r.get("EXPLAIN"),
            "龙虎榜成交额": _num(r.get("BILLBOARD_DEAL_AMT")),
            "占总成交比": _num(r.get("DEAL_AMOUNT_RATIO")),
            "净买入额": _num(r.get("BILLBOARD_NET_AMT")),
            "机构净买入": org_map.get(d, np.nan),
            "解读": r.get("EXPLAIN"),
        })
    for d, v in org_map.items():     # 有机构席位但无榜单主记录的日期补一行
        if all(str(x["date"])[:10] != d for x in recs):
            recs.append({"date": pd.to_datetime(d, errors="coerce"), "上榜原因": "机构席位",
                         "龙虎榜成交额": np.nan, "占总成交比": np.nan,
                         "净买入额": np.nan, "机构净买入": v, "解读": "机构专用席位"})
    df = pd.DataFrame(recs).dropna(subset=["date"]).sort_values("date", ascending=False)
    df = df.reset_index(drop=True)
    _note("龙虎榜来源：东财 RPT_DAILYBILLBOARD_DETAILSNEW + RPT_ORGANIZATION_TRADE_DETAILS"
          "（%d 条）" % len(df))
    return df


def fetch_lhb(symbol, force=False) -> pd.DataFrame:
    """龙虎榜明细（降序）。

    Returns
    -------
    pandas.DataFrame
        ``[date, 上榜原因, 龙虎榜成交额, 占总成交比, 净买入额, 机构净买入, 解读]``；
        失败返回空表。
    """
    code = _code(symbol)
    try:
        if force:
            return _producer_lhb(symbol)
        out = cached(cache_key("lhb", code), config.RUNTIME["cache_ttl_daily"],
                     lambda: _producer_lhb(symbol))
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        _note("fetch_lhb 异常降级：%r" % exc)
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# 5. 北向持股
# --------------------------------------------------------------------------- #
def _producer_northbound(symbol) -> pd.DataFrame:
    """北向持股（东财 ``RPT_MUTUAL_HOLDSTOCKNORTH_STA`` → akshare → 空表）。

    **注意**：该报表带 ``sortColumns=HOLD_DATE`` 会返回 "HOLD_DATE排序列不存在"，
    必须不带排序参数；实测字段 ``TRADE_DATE, HOLD_SHARES, HOLD_MARKET_CAP,
    A_SHARES_RATIO, HOLD_SHARES_RATIO, FREE_SHARES_RATIO, CHANGE_RATE, CLOSE_PRICE``，
    且 600519 仅返回最新 1 行（季度频率）。拿不到属**正常**，不报错。
    """
    rows = _em_query("RPT_MUTUAL_HOLDSTOCKNORTH_STA",
                     '(SECURITY_CODE="%s")' % _code(symbol), page_size=60)
    if rows:
        recs = [{
            "date": pd.to_datetime(str(r.get("TRADE_DATE"))[:10], errors="coerce"),
            "持股股数": _num(r.get("HOLD_SHARES")),
            "持股市值": _num(r.get("HOLD_MARKET_CAP")),
            "持股占比": _num(r.get("HOLD_SHARES_RATIO")),
            "占流通股比": _num(r.get("FREE_SHARES_RATIO")),
            "变动": _num(r.get("CHANGE_RATE")),
            "收盘价": _num(r.get("CLOSE_PRICE")),
        } for r in rows]
        df = pd.DataFrame(recs).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        _note("北向持股来源：东财 RPT_MUTUAL_HOLDSTOCKNORTH_STA（%d 期）" % len(df))
        return df
    try:
        import akshare as ak
        raw = retry_call(lambda: ak.stock_hsgt_individual_em(symbol=_code(symbol)),
                         attempts=2, delay=0.8, default=None, label="hsgt_individual_em")
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            _note("北向持股降级来源：akshare.stock_hsgt_individual_em（列：%s）"
                  % list(raw.columns)[:6])
            return raw
    except Exception as exc:  # noqa: BLE001
        _note("akshare 北向持股失败：%r" % exc)
    _note("北向持股不可得（报表为空 / 已停止披露），返回空表 —— 属正常降级")
    return pd.DataFrame()


def fetch_northbound(symbol, force=False) -> pd.DataFrame:
    """北向持股变动。列见 :func:`_producer_northbound`（``date, 持股股数, 持股市值,
    持股占比, 占流通股比, 变动, 收盘价``）；拿不到返回空表（**不报错**）。"""
    code = _code(symbol)
    try:
        if force:
            return _producer_northbound(symbol)
        out = cached(cache_key("north", code), config.RUNTIME["cache_ttl_daily"],
                     lambda: _producer_northbound(symbol))
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        _note("fetch_northbound 异常降级：%r" % exc)
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# 6. 十大流通股东
# --------------------------------------------------------------------------- #
def _producer_top10(symbol) -> pd.DataFrame:
    """十大流通股东（东财 ``RPT_F10_EH_FREEHOLDERS``，实测可用）。

    字段：``END_DATE, HOLDER_NAME, HOLD_NUM, FREE_HOLDNUM_RATIO(占流通股比%),
    HOLD_NUM_CHANGE(数值或"不变"/"新进"), CHANGE_RATIO, HOLDER_RANK, HOLDER_TYPE``。
    ``df.attrs["summary"]`` 附最近两期增减持汇总。
    """
    rows = _em_query("RPT_F10_EH_FREEHOLDERS", '(SECUCODE="%s")' % _secucode(symbol),
                     page_size=60, sort="END_DATE")
    if not rows:
        return pd.DataFrame()
    recs = []
    for r in rows:
        chg = r.get("HOLD_NUM_CHANGE")
        if isinstance(chg, str):
            chg_num = 0.0 if chg.strip() in ("不变", "新进", "") else _num(chg)
        else:
            chg_num = _num(chg)
        recs.append({
            "报告期": pd.to_datetime(str(r.get("END_DATE"))[:10], errors="coerce"),
            "股东名称": r.get("HOLDER_NAME"),
            "持股数": _num(r.get("HOLD_NUM")),
            "占流通股比": _num(r.get("FREE_HOLDNUM_RATIO")),
            "持股变动": chg_num,
            "变动比例": _num(r.get("CHANGE_RATIO")),
            "排名": _num(r.get("HOLDER_RANK")),
            "类型": r.get("HOLDER_TYPE"),
            "变动说明": chg if isinstance(chg, str) else "",
        })
    df = pd.DataFrame(recs).dropna(subset=["报告期"])
    df = df.sort_values(["报告期", "排名"], ascending=[False, True]).reset_index(drop=True)
    # ---- 最近两期增减持汇总 --------------------------------------------- #
    try:
        periods = sorted(df["报告期"].unique(), reverse=True)[:2]
        summary = {"periods": [str(p)[:10] for p in periods], "total_change": 0.0,
                   "n_increase": 0, "n_decrease": 0, "detail": ""}
        if periods:
            cur = df[df["报告期"] == periods[0]]
            chg = pd.to_numeric(cur["持股变动"], errors="coerce").fillna(0.0)
            summary["total_change"] = float(chg.sum())
            summary["n_increase"] = int((chg > 0).sum())
            summary["n_decrease"] = int((chg < 0).sum())
            summary["detail"] = ("最近报告期 %s：十大流通股东合计%s %.2f 万股"
                                 "（增持 %d 家 / 减持 %d 家）"
                                 % (str(periods[0])[:10],
                                    "增持" if summary["total_change"] >= 0 else "减持",
                                    abs(summary["total_change"]) / 1e4,
                                    summary["n_increase"], summary["n_decrease"]))
        df.attrs["summary"] = summary
    except Exception as exc:  # noqa: BLE001
        log.debug("十大股东汇总失败: %r", exc)
    _note("十大流通股东来源：东财 RPT_F10_EH_FREEHOLDERS（%d 条 / %d 期）"
          % (len(df), df["报告期"].nunique()))
    return df


def fetch_top10_float_holders(symbol, force=False) -> pd.DataFrame:
    """前十大流通股东明细 + 最近两期增减持汇总（``df.attrs["summary"]``）。"""
    code = _code(symbol)
    try:
        if force:
            return _producer_top10(symbol)
        out = cached(cache_key("top10", code), config.RUNTIME["cache_ttl_daily"],
                     lambda: _producer_top10(symbol))
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        _note("fetch_top10_float_holders 异常降级：%r" % exc)
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# 7. 快照（供控盘度使用；本模块自带解析，避免与其它模块耦合）
# --------------------------------------------------------------------------- #
def _producer_snapshot(symbol) -> dict:
    """腾讯实时快照（GBK，字段索引实测固化于 :data:`_TX_IDX`）。"""
    txt = http_get(_TENCENT_URL % _prefix(symbol), encoding="gbk")
    if not txt or '="' not in txt:
        return {}
    parts = txt.split('="', 1)[1].rsplit('"', 1)[0].split("~")
    out = {}
    for k, i in _TX_IDX.items():
        if i < len(parts):
            out[k] = parts[i] if k == "name" else _num(parts[i])
    if "market_cap_yi" in out:
        out["market_cap"] = out["market_cap_yi"] * 1e8
    if "float_market_cap_yi" in out:
        out["float_market_cap"] = out["float_market_cap_yi"] * 1e8
    out["source"] = "tencent"
    return out


def fetch_snapshot(symbol, force=False) -> dict:
    """个股实时快照（价/换手/量比/市值/PE/PB）；失败返回 ``{}``。"""
    code = _code(symbol)
    try:
        if force:
            return _producer_snapshot(symbol)
        out = cached(cache_key("snap", code), config.RUNTIME["cache_ttl_seconds"],
                     lambda: _producer_snapshot(symbol))
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001
        _note("腾讯快照失败：%r" % exc)
        return {}


# --------------------------------------------------------------------------- #
# 8. 控盘度 / 主力行为
# --------------------------------------------------------------------------- #
def control_degree(symbol, frame=None, holders=None) -> dict:
    """控盘程度评分（0~1）。

    评分逻辑（三项加权，权重写死在下方 ``w_*``）：

    * **换手率（35%）**：近 20 日均换手率越低越像高度控盘（≥8% 记 0 分，≤1% 记满分）。
    * **股东户数（30%）**：户均持股增加 = 股东户数环比下降 = 筹码集中（环比 −10% 记满分，
      +10% 记 0 分）。
    * **前十大流通股东占比（35%）**：≥65% 记满分，≤20% 记 0 分（参考
      ``config.STAGE_RULES["pricing_power_top10_pct"]=55`` 为机构定价门槛）。

    Returns
    -------
    dict
        ``{"score","turnover_avg","turnover_trend","holder_num_change_pct","top10_ratio",
        "float_cap","level","detail"}``；数据不足时 ``score=0``、``level="数据不足"``。
    """
    out = {"score": 0.0, "turnover_avg": float("nan"), "turnover_trend": float("nan"),
           "holder_num_change_pct": float("nan"), "top10_ratio": float("nan"),
           "float_cap": float("nan"), "level": "数据不足", "detail": ""}
    try:
        snap = fetch_snapshot(symbol)
        # ---- 换手率 ------------------------------------------------------ #
        turns = []
        f = _ensure_frame(frame)
        if f is not None and "turnover" in f.columns:
            turns = [float(x) for x in pd.to_numeric(f["turnover"], errors="coerce")
                     .dropna().tail(20).tolist()]
        if not turns:
            t = _num((snap or {}).get("turnover"))
            if not math.isnan(t):
                turns = [t]
        turnover_avg = float(np.mean(turns)) if turns else float("nan")
        turnover_trend = float("nan")
        if len(turns) >= 10:
            recent, base = np.mean(turns[-5:]), np.mean(turns[-10:-5])
            if base > 0:
                turnover_trend = (recent / base - 1.0) * 100.0
        c_turn = (1.0 - clamp(turnover_avg / 8.0, 0.0, 1.0)) if not math.isnan(turnover_avg) \
            else float("nan")
        # ---- 股东户数 ---------------------------------------------------- #
        h = holders if isinstance(holders, pd.DataFrame) else fetch_holders(symbol)
        chg_pct = float("nan")
        if isinstance(h, pd.DataFrame) and not h.empty and "股东户数环比" in h.columns:
            try:
                chg_pct = float(pd.to_numeric(h["股东户数环比"], errors="coerce")
                                .dropna().iloc[0])
            except Exception:  # noqa: BLE001
                chg_pct = float("nan")
        c_holder = clamp(0.5 - chg_pct / 20.0, 0.0, 1.0) if not math.isnan(chg_pct) \
            else float("nan")
        # ---- 前十大流通股东占比 ------------------------------------------ #
        t10 = fetch_top10_float_holders(symbol)
        ratio = float("nan")
        if isinstance(t10, pd.DataFrame) and not t10.empty:
            try:
                latest = t10["报告期"].max()
                ratio = float(pd.to_numeric(
                    t10[t10["报告期"] == latest]["占流通股比"], errors="coerce").sum())
            except Exception:  # noqa: BLE001
                ratio = float("nan")
        c_top = clamp((ratio - 20.0) / 45.0, 0.0, 1.0) if not math.isnan(ratio) else float("nan")
        # ---- 加权（缺失项按剩余权重归一化） ------------------------------ #
        w_turn, w_hold, w_top = 0.35, 0.30, 0.35
        pairs = [(c_turn, w_turn), (c_holder, w_hold), (c_top, w_top)]
        valid = [(v, w) for v, w in pairs if not math.isnan(v)]
        score = 0.0
        if valid:
            tot = sum(w for _, w in valid)
            score = sum(v * w for v, w in valid) / tot
        level = "数据不足"
        if valid:
            level = "高度控盘" if score >= 0.7 else ("中度控盘" if score >= 0.45 else "分散")
        parts = []
        if not math.isnan(turnover_avg):
            parts.append("20 日均换手 %.2f%%%s" % (
                turnover_avg,
                "（换手持续放大 %.1f%%）" % turnover_trend
                if not math.isnan(turnover_trend) else ""))
        if not math.isnan(chg_pct):
            parts.append("股东户数环比 %+.1f%%（%s）"
                         % (chg_pct, "筹码集中" if chg_pct < 0 else "筹码分散"))
        if not math.isnan(ratio):
            parts.append("前十大流通股东合计 %.1f%%" % ratio)
        if not valid:
            parts.append("缺少换手率/股东户数/十大股东数据，无法评估控盘度")
        out.update({
            "score": round(float(score), 3), "turnover_avg": turnover_avg,
            "turnover_trend": turnover_trend, "holder_num_change_pct": chg_pct,
            "top10_ratio": ratio,
            "float_cap": _num((snap or {}).get("float_market_cap")),
            "level": level,
            "detail": "；".join(parts) + ("；控盘评分 %.2f（换手 35%%、股东户数 30%%、"
                                          "十大股东 35%%）" % score if valid else ""),
        })
    except Exception as exc:  # noqa: BLE001
        out["detail"] = "控盘度计算异常，已降级：%r" % exc
        _note(out["detail"])
    return out


def main_force_profile(flow, frame=None) -> dict:
    """主力行为画像：**吃货 / 洗盘 / 吐货 / 出货 / 观望**。

    判定逻辑（阈值见 :data:`FORCE_TH`，全部写进注释便于校准）::

        net_ratio_20 = Σ主力净额(20日) / Σ|主力净额|(20日)   ∈ [-1,1]  方向主导度
        price_20d_chg = 20 日涨跌幅（%，取自 flow.close，缺失则用 frame）

        net_ratio_20 > +0.15 且 涨幅 < 8%          → 吃货（低调吸筹，最健康）
        net_ratio_20 > +0.15 且 涨幅 ≥ 8%          → 吃货（拉高建仓，可信度打折）
        net_ratio_20 < -0.15 且 涨幅 > -3%         → 洗盘（资金流出但股价抗跌）
        net_ratio_20 < -0.15 且 -8% ≤ 涨幅 ≤ -3%   → 吐货（资金撤、股价走弱）
        net_ratio_20 < -0.15 且 涨幅 < -8%         → 出货（破位下跌，主力出逃）
        其它                                        → 观望

    Returns
    -------
    dict
        ``{"behavior","confidence","net_5d","net_20d","price_20d_chg","consistency",
        "detail"}``；``flow`` 为空时返回观望 0 置信度。
    """
    out = {"behavior": "观望", "confidence": 0.0, "net_5d": float("nan"),
           "net_20d": float("nan"), "price_20d_chg": float("nan"),
           "consistency": float("nan"), "detail": "无资金流数据"}
    try:
        df = _ensure_frame(flow)
        if df is None or "main_net" not in df.columns:
            return out
        df = df.copy()
        df["main_net"] = pd.to_numeric(df["main_net"], errors="coerce")
        df = df.dropna(subset=["main_net"]).tail(20)
        if df.empty:
            return out
        net20 = float(df["main_net"].sum())
        net5 = float(df["main_net"].tail(5).sum())
        denom = float(df["main_net"].abs().sum())
        ratio = net20 / denom if denom > 0 else 0.0
        # 20 日涨跌幅：优先用资金流表里的收盘价，其次用 K 线
        chg = float("nan")
        closes = pd.to_numeric(df.get("close"), errors="coerce").dropna() \
            if "close" in df.columns else pd.Series(dtype=float)
        if len(closes) >= 2 and float(closes.iloc[0]) > 0:
            chg = (float(closes.iloc[-1]) / float(closes.iloc[0]) - 1.0) * 100.0
        else:
            f = _ensure_frame(frame)
            if f is not None and "close" in f.columns:
                cl = pd.to_numeric(f["close"], errors="coerce").dropna().tail(21)
                if len(cl) >= 2 and float(cl.iloc[0]) > 0:
                    chg = (float(cl.iloc[-1]) / float(cl.iloc[0]) - 1.0) * 100.0
        # 一致性：近 20 日「主力净额方向」与「当日涨跌方向」同号的比例
        consistency = float("nan")
        if "pct_chg" in df.columns:
            pc = pd.to_numeric(df["pct_chg"], errors="coerce")
            mask = pc.notna() & (df["main_net"] != 0) & (pc != 0)
            if mask.sum() >= 5:
                same = (np.sign(df.loc[mask, "main_net"]) == np.sign(pc[mask])).mean()
                consistency = float(same)
        # ---- 判定 -------------------------------------------------------- #
        th = FORCE_TH
        behavior, conf, why = "观望", 0.15, "资金与价格信号不明确，建议观望"
        if not math.isnan(chg):
            if ratio > th["strong_net"]:
                if chg < th["mild_up"]:
                    behavior, conf = "吃货", clamp(0.45 + abs(ratio) * 0.8, 0.3, 0.9)
                    why = "连续多日主力净流入（%.2f 亿）而股价仅 %+.1f%%，属低调吸筹" \
                          % (net20 / 1e8, chg)
                else:
                    behavior, conf = "吃货", clamp(0.3 + abs(ratio) * 0.5, 0.2, 0.6)
                    why = "主力净流入且股价已 %+.1f%%，可能是拉高建仓/追涨资金，需防高位" % chg
            elif ratio < -th["strong_net"]:
                if chg > th["resist_drop"]:
                    behavior, conf = "洗盘", clamp(0.4 + abs(ratio) * 0.7, 0.25, 0.8)
                    why = "主力净流出 %.2f 亿但股价 %+.1f%% 抗跌，更像洗盘而非出货" \
                          % (net20 / 1e8, chg)
                elif chg >= th["breakdown"]:
                    behavior, conf = "吐货", clamp(0.4 + abs(ratio) * 0.7, 0.25, 0.8)
                    why = "主力净流出 %.2f 亿且股价 %+.1f%%，主力在派发" % (net20 / 1e8, chg)
                else:
                    behavior, conf = "出货", clamp(0.5 + abs(ratio) * 0.8, 0.3, 0.9)
                    why = "主力大幅净流出 %.2f 亿且股价破位 %+.1f%%，属出货" \
                          % (net20 / 1e8, chg)
        if not math.isnan(consistency):
            conf = clamp(conf * (0.6 + 0.4 * consistency), 0.05, 0.95)
        out.update({"behavior": behavior, "confidence": round(float(conf), 3),
                    "net_5d": net5, "net_20d": net20, "price_20d_chg": chg,
                    "consistency": consistency,
                    "detail": "%s；%s（20 日主力净额 %+.2f 亿，方向主导度 %.2f%s）"
                              % (behavior, why, net20 / 1e8, ratio,
                                 "，量价同向比例 %.0f%%" % (consistency * 100)
                                 if not math.isnan(consistency) else "")})
    except Exception as exc:  # noqa: BLE001
        out["detail"] = "主力行为判定异常，已降级：%r" % exc
        _note(out["detail"])
    return out


# --------------------------------------------------------------------------- #
# 9. 因子合成
# --------------------------------------------------------------------------- #
def _flow_factors(flow) -> list:
    fs: list = []
    df = _ensure_frame(flow)
    if df is None or "main_net" not in df.columns:
        fs.append(_neutral("capital_main_net", "主力净流入", "资金流数据缺失", 0.15))
        fs.append(_neutral("capital_super_big", "超大单/大单结构", "资金流数据缺失", 0.15))
        fs.append(_neutral("capital_main_ratio", "主力净占比", "资金流数据缺失", 0.15))
        fs.append(_neutral("capital_trend_consistency", "主力连续净流入", "资金流数据缺失", 0.15))
        fs.append(_neutral("capital_divergence", "量价资金背离", "资金流数据缺失", 0.15))
        return fs
    df = df.copy()
    for c in ("main_net", "main_pct", "big_net", "super_net", "close", "pct_chg"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    n = len(df)
    rel_base = 0.9 if n >= 60 else (0.7 if n >= 20 else 0.45)

    # ---- 1) 主力净流入（5/10/20 日加权：0.5/0.3/0.2） -------------------- #
    def _sum(col, w):
        try:
            return float(df[col].dropna().tail(w).sum())
        except Exception:  # noqa: BLE001
            return float("nan")
    n5, n10, n20 = _sum("main_net", 5), _sum("main_net", 10), _sum("main_net", 20)
    scale = float(df["main_net"].abs().tail(20).mean() or 0.0) * 20 or 1.0
    score = sum(w * (v / scale if not math.isnan(v) else 0.0)
                for v, w in ((n5, 0.5), (n10, 0.3), (n20, 0.2)))
    fs.append(Factor("capital_main_net", "主力净流入", Category.CAPITAL,
                     DIR_BULL if score > 0.05 else (DIR_BEAR if score < -0.05 else 0),
                     clamp(abs(score), 0, 1), clamp(rel_base * 0.95, 0.1, 0.95),
                     value={"净流入5日": n5, "净流入10日": n10, "净流入20日": n20},
                     detail="近 5/10/20 日主力资金净流入 %+.2f / %+.2f / %+.2f 亿元"
                            "（加权口径 0.5/0.3/0.2）"
                            % (n5 / 1e8, n10 / 1e8, n20 / 1e8),
                     theory="资金流向理论", horizon=FLOW_HORIZON))

    # ---- 2) 超大单 / 大单结构 -------------------------------------------- #
    s5, b5 = _sum("super_net", 5), _sum("big_net", 5)
    tot = (0.0 if math.isnan(s5) else s5) + (0.0 if math.isnan(b5) else b5)
    # 归一化基准：近 20 日「超大单+大单」平均规模 × 5 日（避免恒为 ±1）
    try:
        base_sb = float((df["super_net"].abs() + df["big_net"].abs())
                        .tail(20).mean() or 0.0) * 5.0
    except Exception:  # noqa: BLE001
        base_sb = 0.0
    base_sb = base_sb or 1.0
    s_score = tot / base_sb
    fs.append(Factor("capital_super_big", "超大单/大单结构", Category.CAPITAL,
                     DIR_BULL if s_score > 0.1 else (DIR_BEAR if s_score < -0.1 else 0),
                     clamp(abs(s_score) / 2.0, 0, 1), clamp(rel_base * 0.85, 0.1, 0.9),
                     value={"超大单5日": s5, "大单5日": b5, "相对规模": s_score},
                     detail="近 5 日超大单 %+.2f 亿、大单 %+.2f 亿，%s"
                            % (s5 / 1e8, b5 / 1e8,
                               "机构/大资金主导买入" if s_score > 0.1 else
                               ("大资金主导卖出" if s_score < -0.1 else "大单多空均衡")),
                     theory="大单资金结构", horizon=FLOW_HORIZON))

    # ---- 3) 主力净占比 ---------------------------------------------------- #
    pct = df["main_pct"].dropna().tail(5)
    mp = float(pct.mean()) if len(pct) else float("nan")
    fs.append(Factor("capital_main_ratio", "主力净占比", Category.CAPITAL,
                     DIR_BULL if (not math.isnan(mp) and mp > 0.5) else
                     (DIR_BEAR if (not math.isnan(mp) and mp < -0.5) else 0),
                     clamp(abs(mp) / 6.0, 0, 1) if not math.isnan(mp) else 0.0,
                     clamp(rel_base * 0.8 if not math.isnan(mp) else 0.15, 0.05, 0.9),
                     value=mp,
                     detail=("近 5 日主力净占比均值 %+.2f%%（占成交额），%s"
                             % (mp, "资金持续净流入" if mp > 0 else "资金持续净流出"))
                     if not math.isnan(mp) else "资金流缺少净占比字段",
                     theory="资金流向理论", horizon=FLOW_HORIZON))

    # ---- 4) 连续净流入天数 ------------------------------------------------ #
    seq, streak = df["main_net"].dropna().tolist(), 0
    for v in reversed(seq):
        if v > 0:
            streak += 1
        else:
            break
    out_streak = 0
    for v in reversed(seq):
        if v < 0:
            out_streak += 1
        else:
            break
    best = streak if streak >= out_streak else -out_streak
    fs.append(Factor("capital_trend_consistency", "主力连续净流入", Category.CAPITAL,
                     DIR_BULL if best > 0 else (DIR_BEAR if best < 0 else 0),
                     clamp(abs(best) / 8.0, 0, 1), clamp(rel_base * 0.8, 0.1, 0.9),
                     value=best,
                     detail=("主力资金已连续 %d 日净流入" % best) if best > 0 else
                            (("主力资金已连续 %d 日净流出" % abs(best)) if best < 0
                             else "最近一日主力资金无明确方向"),
                     theory="资金流向理论", horizon=FLOW_HORIZON))

    # ---- 5) 量价资金背离 -------------------------------------------------- #
    pc = df["pct_chg"].dropna().tail(5)
    price_5 = float(pc.sum()) if len(pc) else float("nan")
    div_dir, div_str, div_detail = 0, 0.0, "近 5 日资金与价格方向一致，无背离"
    if not math.isnan(price_5) and not math.isnan(n5):
        if price_5 > 1.0 and n5 < 0:
            div_dir, div_str = DIR_BEAR, clamp(abs(n5) / scale, 0.2, 0.9)
            div_detail = ("股价 5 日 %+.1f%% 但主力净流出 %.2f 亿，价涨资金撤，"
                          "警惕拉高出货" % (price_5, abs(n5) / 1e8))
        elif price_5 < -1.0 and n5 > 0:
            div_dir, div_str = DIR_BULL, clamp(abs(n5) / scale, 0.2, 0.9)
            div_detail = ("股价 5 日 %+.1f%% 但主力净流入 %.2f 亿，价跌资金进，"
                          "疑似吸筹/洗盘" % (price_5, n5 / 1e8))
    fs.append(Factor("capital_divergence", "量价资金背离", Category.CAPITAL, div_dir,
                     div_str, clamp(rel_base * 0.75, 0.1, 0.85),
                     value={"5日涨跌幅": price_5, "5日主力净额": n5}, detail=div_detail,
                     theory="量价关系 / 资金背离", horizon=FLOW_HORIZON))
    return fs


def _lhb_factors(lhb) -> list:
    df = lhb if isinstance(lhb, pd.DataFrame) else None
    if df is None or df.empty:
        return [_neutral("lhb_institution", "龙虎榜机构席位",
                         "无龙虎榜记录（或数据不可得），资金面无席位信号", 0.35,
                         theory="龙虎榜席位分析", horizon=("d1", "d5", "d10", "d20"))]
    try:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        latest = d.dropna(subset=["date"]).sort_values("date").iloc[-1]
        days_ago = (now_cn().date() - latest["date"].date()).days
        net = _num(latest.get("机构净买入"))
        if days_ago > 90 or math.isnan(net):
            return [_neutral("lhb_institution", "龙虎榜机构席位",
                             "最近上榜 %d 天前（%s），时效性不足，不计入方向"
                             % (days_ago, str(latest["date"])[:10]), 0.3,
                             theory="龙虎榜席位分析",
                             horizon=("d1", "d5", "d10", "d20"))]
        strength = clamp(abs(net) / 5e8, 0.2, 0.9)
        return [Factor("lhb_institution", "龙虎榜机构席位", Category.CAPITAL,
                       DIR_BULL if net > 0 else DIR_BEAR, strength,
                       clamp(0.75 - days_ago / 120.0, 0.3, 0.85),
                       value=net,
                       detail="%s 龙虎榜机构席位净%s %.2f 亿元（%s）"
                              % (str(latest["date"])[:10], "买入" if net > 0 else "卖出",
                                 abs(net) / 1e8, latest.get("上榜原因") or ""),
                       theory="龙虎榜席位分析", horizon=("d1", "d5", "d10", "d20"))]
    except Exception as exc:  # noqa: BLE001
        _note("龙虎榜因子异常：%r" % exc)
        return [_neutral("lhb_institution", "龙虎榜机构席位", "龙虎榜解析异常", 0.2)]


def _north_factors(north) -> list:
    df = north if isinstance(north, pd.DataFrame) else None
    if df is None or df.empty:
        return [_neutral("northbound_change", "北向资金",
                         "北向持股数据不可得（报表为空/已停止披露）", 0.2,
                         theory="外资定价权", horizon=("d5", "d10", "d20", "m1"))]
    try:
        col = "持股股数" if "持股股数" in df.columns else df.columns[1]
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 2:
            return [_neutral("northbound_change", "北向资金", "北向持股样本不足", 0.25,
                             theory="外资定价权", horizon=("d5", "d10", "d20", "m1"))]
        chg = (float(s.iloc[-1]) / float(s.iloc[-2]) - 1.0) * 100.0 if float(s.iloc[-2]) else 0.0
        ratio = _num(df.iloc[-1].get("占流通股比")) if "占流通股比" in df.columns else float("nan")
        return [Factor("northbound_change", "北向资金", Category.CAPITAL,
                       DIR_BULL if chg > 0.5 else (DIR_BEAR if chg < -0.5 else 0),
                       clamp(abs(chg) / 15.0, 0.1, 0.85), 0.55,
                       value=chg,
                       detail="北向持股环比 %+.2f%%（最新 %s，占流通股 %s）"
                              % (chg, str(df.iloc[-1].get("date"))[:10],
                                 "%.2f%%" % ratio if not math.isnan(ratio) else "—"),
                       theory="外资定价权", horizon=("d5", "d10", "d20", "m1"))]
    except Exception as exc:  # noqa: BLE001
        _note("北向因子异常：%r" % exc)
        return [_neutral("northbound_change", "北向资金", "北向数据处理异常", 0.2)]


def _holder_factors(holders, top10) -> list:
    fs: list = []
    h = holders if isinstance(holders, pd.DataFrame) else None
    if h is None or h.empty or "股东户数环比" not in h.columns:
        fs.append(_neutral("holder_concentration", "筹码集中度（股东户数）",
                           "无股东户数数据", 0.2, theory="筹码分布"))
    else:
        try:
            d = h.copy()
            d["股东户数环比"] = pd.to_numeric(d["股东户数环比"], errors="coerce")
            chg = float(d["股东户数环比"].dropna().iloc[0])
            avg = _num(d.iloc[0].get("户均持股数"))
            score = clamp(-chg / 15.0, -1, 1)     # 户数减少 = 筹码集中 = 看涨
            fs.append(Factor("holder_concentration", "筹码集中度（股东户数）",
                             Category.CAPITAL,
                             DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.1 else 0),
                             clamp(abs(score), 0, 1), 0.65, value=chg,
                             detail="最新股东户数环比 %+.1f%%（%s），户均持股 %s 股"
                                    % (chg, "户数减少、筹码集中" if chg < 0
                                       else "户数增加、筹码分散",
                                       "%.0f" % avg if not math.isnan(avg) else "—"),
                             theory="筹码分布 / 行为金融学"))
        except Exception as exc:  # noqa: BLE001
            _note("股东户数因子异常：%r" % exc)
            fs.append(_neutral("holder_concentration", "筹码集中度（股东户数）",
                               "股东户数解析异常", 0.2))
    t = top10 if isinstance(top10, pd.DataFrame) else None
    if t is None or t.empty or "持股变动" not in t.columns:
        fs.append(_neutral("top10_float_change", "前十大流通股东增减持",
                           "无前十大流通股东数据", 0.2, theory="机构持仓分析",
                           horizon=("d20", "m1", "q1", "h1")))
    else:
        try:
            summary = (t.attrs or {}).get("summary") or {}
            total = _num(summary.get("total_change"))
            if math.isnan(total):
                latest = t["报告期"].max()
                total = float(pd.to_numeric(
                    t[t["报告期"] == latest]["持股变动"], errors="coerce").fillna(0).sum())
            base = float(pd.to_numeric(t["持股数"], errors="coerce").fillna(0).sum()) or 1.0
            score = clamp(total / base * 20.0, -1, 1)
            fs.append(Factor("top10_float_change", "前十大流通股东增减持", Category.CAPITAL,
                             DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.1 else 0),
                             clamp(abs(score), 0, 1), 0.65, value=total,
                             detail=summary.get("detail") or
                             "前十大流通股东合计变动 %.2f 万股" % (total / 1e4),
                             theory="机构持仓分析", horizon=("d20", "m1", "q1", "h1")))
        except Exception as exc:  # noqa: BLE001
            _note("十大流通股东因子异常：%r" % exc)
            fs.append(_neutral("top10_float_change", "前十大流通股东增减持",
                               "十大股东解析异常", 0.2))
    return fs


def _margin_factors(margin) -> list:
    df = margin if isinstance(margin, pd.DataFrame) else None
    if df is None or df.empty or "融资余额" not in df.columns:
        return [_neutral("margin_balance_trend", "两融余额趋势", "无两融数据", 0.2,
                         theory="杠杆资金", horizon=("d5", "d10", "d20", "m1"))]
    try:
        s = pd.to_numeric(df["融资余额"], errors="coerce").dropna()
        if len(s) < 5:
            return [_neutral("margin_balance_trend", "两融余额趋势", "两融样本不足", 0.25,
                             theory="杠杆资金", horizon=("d5", "d10", "d20", "m1"))]
        chg5 = (float(s.iloc[-1]) / float(s.iloc[-6]) - 1.0) * 100.0 if float(s.iloc[-6]) else 0.0
        chg20 = (float(s.iloc[-1]) / float(s.iloc[0]) - 1.0) * 100.0 if float(s.iloc[0]) else 0.0
        ratio = _num(df.iloc[-1].get("融资余额占流通市值比"))
        score = clamp(chg5 / 8.0, -1, 1) * 0.6 + clamp(chg20 / 15.0, -1, 1) * 0.4
        return [Factor("margin_balance_trend", "两融余额趋势", Category.CAPITAL,
                       DIR_BULL if score > 0.08 else (DIR_BEAR if score < -0.08 else 0),
                       clamp(abs(score), 0, 1), 0.6,
                       value={"5日": chg5, "20日": chg20},
                       detail="融资余额 5 日 %+.2f%%、区间 %+.2f%%（占流通市值 %s），%s"
                              % (chg5, chg20,
                                 "%.2f%%" % ratio if not math.isnan(ratio) else "—",
                                 "杠杆资金加仓" if score > 0.08 else
                                 ("杠杆资金撤离" if score < -0.08 else "杠杆资金平稳")),
                       theory="杠杆资金 / 两融余额", horizon=("d5", "d10", "d20", "m1"))]
    except Exception as exc:  # noqa: BLE001
        _note("两融因子异常：%r" % exc)
        return [_neutral("margin_balance_trend", "两融余额趋势", "两融数据解析异常", 0.2)]


def capital_factors(symbol, flow=None, holders=None, margin=None, lhb=None, north=None,
                    snapshot=None, frame=None, top10=None) -> list:
    """产出资金面因子列表（详见模块 docstring 的 id 清单）。

    * 资金流类因子 ``horizon=("d1","d5","d10","d20","m1")``（短线主导）；
      ``control_degree`` 为全周期（``horizon=None``）。
    * 任一参数缺失都不抛异常：对应因子退化为**中性 + 低可信度**，
      ``fund_holding_change`` 在没有基金持仓数据时**直接跳过**（按要求）。
    * 传入的 DataFrame **不会被修改**。
    * ``top10``（可选，:func:`fetch_top10_float_holders` 的产出）用于
      ``top10_float_change``；不传则该因子退化为中性（**因子生成本身不联网**）。

    Returns
    -------
    list[Factor]
    """
    fs: list = []
    try:
        fs.extend(_flow_factors(flow))
        fs.extend(_lhb_factors(lhb))
        fs.extend(_north_factors(north))
        fs.extend(_holder_factors(holders, top10))
        fs.extend(_margin_factors(margin))
        # ---- 基金持仓变动：无稳定免费数据源 → 按要求跳过 ------------------ #
        fund_holding = None
        if isinstance(snapshot, dict):
            fund_holding = snapshot.get("fund_holding_change")
        if fund_holding is not None:
            try:
                chg = float(fund_holding)
                fs.append(Factor("fund_holding_change", "基金持仓变动", Category.CAPITAL,
                                 DIR_BULL if chg > 0 else (DIR_BEAR if chg < 0 else 0),
                                 clamp(abs(chg) / 20.0, 0.1, 0.8), 0.5, value=chg,
                                 detail="基金持仓变动 %+.1f%%" % chg,
                                 theory="机构持仓分析", horizon=("q1", "h1", "y1")))
            except Exception:  # noqa: BLE001
                pass
        # ---- 控盘度 ------------------------------------------------------- #
        cd = control_degree(symbol, frame=frame, holders=holders)
        score = _num(cd.get("score"), 0.0)
        level = cd.get("level", "数据不足")
        if level == "数据不足":
            fs.append(_neutral("control_degree", "控盘程度", cd.get("detail") or "数据不足",
                               0.2, value=None, theory="筹码分布 / 控盘理论", horizon=None))
        else:
            fs.append(Factor("control_degree", "控盘程度", Category.CAPITAL,
                             DIR_BULL if score > 0.6 else (DIR_BEAR if score < 0.35 else 0),
                             clamp(abs(score - 0.5) * 2.0, 0, 1), 0.6,
                             value="%s(%.2f)" % (level, score), detail=cd.get("detail", ""),
                             theory="筹码分布 / 控盘理论", horizon=None))
        # ---- 主力行为 ----------------------------------------------------- #
        prof = main_force_profile(flow, frame=frame)
        beh = prof.get("behavior", "观望")
        dir_map = {"吃货": DIR_BULL, "洗盘": DIR_BULL, "吐货": DIR_BEAR,
                   "出货": DIR_BEAR, "观望": DIR_NEUTRAL}
        strength_map = {"吃货": 0.7, "洗盘": 0.35, "吐货": 0.6, "出货": 0.8, "观望": 0.1}
        fs.append(Factor("main_force_behavior", "主力行为画像", Category.CAPITAL,
                         dir_map.get(beh, DIR_NEUTRAL), strength_map.get(beh, 0.1),
                         clamp(_num(prof.get("confidence"), 0.2), 0.05, 0.9),
                         value=beh, detail=prof.get("detail", ""),
                         theory="主力行为学 / 量价关系", horizon=FLOW_HORIZON))
    except Exception as exc:  # noqa: BLE001
        _note("capital_factors 内部异常（已降级）：%r" % exc)
        fs.append(_neutral("capital_main_net", "主力净流入", "资金面因子生成异常，已降级", 0.1))
    return fs


# --------------------------------------------------------------------------- #
# 自测
# --------------------------------------------------------------------------- #
def _fake_flow(n: int = 150, seed: int = 7) -> pd.DataFrame:
    """合成资金流表（含滚动列），用于离线自测。"""
    import random
    rng = random.Random(seed)
    rows, price = [], 30.0
    for i in range(n):
        px_chg = rng.gauss(0.0008, 0.018)
        price *= 1 + px_chg
        main = rng.gauss(6e6, 4.0e7)
        sup, big = main * 0.6, main * 0.4
        mid, small = -main * 0.35, -main * 0.65
        amt = abs(main) * 6 + 1e8
        rows.append({"date": pd.Timestamp("2025-01-02") + pd.Timedelta(days=i),
                     "main_net": main, "small_net": small, "mid_net": mid, "big_net": big,
                     "super_net": sup, "main_pct": main / amt * 100,
                     "big_pct": big / amt * 100, "super_pct": sup / amt * 100,
                     "close": round(price, 2), "pct_chg": px_chg * 100})
    return _finalize_flow(pd.DataFrame(rows))


def _fake_holders() -> pd.DataFrame:
    return pd.DataFrame({
        "截止日": pd.to_datetime(["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"]),
        "股东户数": [296404, 243159, 250100, 262000],
        "户均持股数": [4217.5, 5140.0, 4998.0, 4771.0],
        "户均持股金额": [5.0e6, 6.1e6, 5.8e6, 5.5e6],
        "股东户数环比": [21.9, -2.8, -4.5, 1.2],
        "股东户数同比": [12.5, -6.0, -8.0, 3.0],
    })


def _fake_margin(days: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2026-07-01", periods=days, freq="B")
    base = np.linspace(1.55e10, 1.72e10, days)
    return pd.DataFrame({"date": idx, "融资余额": base,
                         "融资买入额": base * 0.02, "融券余量": 2.0e5,
                         "融券余额": 2.5e8, "融资融券余额": base * 1.015,
                         "融资余额占流通市值比": 1.05, "融资净买入": np.gradient(base),
                         "收盘价": np.linspace(1200, 1263, days), "涨跌幅": 0.1})


def _fake_lhb() -> pd.DataFrame:
    return pd.DataFrame([{"date": pd.Timestamp("2026-09-20"), "上榜原因": "日涨幅偏离值达7%",
                          "龙虎榜成交额": 3.2e8, "占总成交比": 12.5, "净买入额": 1.4e8,
                          "机构净买入": 2.1e8, "解读": "2家机构买入"}])


def _fake_top10() -> pd.DataFrame:
    df = pd.DataFrame({
        "报告期": pd.to_datetime(["2026-06-30"] * 4),
        "股东名称": ["集团", "证金", "香港中央结算", "某基金"],
        "持股数": [6.81e8, 1.2e8, 5.3e7, 2.0e7],
        "占流通股比": [54.5, 9.6, 4.3, 1.6],
        "持股变动": [1.2e6, -3.0e6, 2.5e6, 8.0e5],
        "变动比例": [0.18, -2.4, 4.9, 4.2],
        "排名": [1, 2, 3, 4], "类型": ["其它", "投资公司", "其它", "基金"],
    })
    total = float(df["持股变动"].sum())
    df.attrs["summary"] = {"periods": ["2026-06-30"], "total_change": total,
                           "n_increase": int((df["持股变动"] > 0).sum()),
                           "n_decrease": int((df["持股变动"] < 0).sum()),
                           "detail": "最近报告期 2026-06-30：十大流通股东合计增持 1.50 万股"}
    return df


def _self_test() -> None:
    """离线自测：合成数据 + monkeypatch 掉 fetch 函数，**不发网络请求**。"""
    import core.selftest_data as sd

    print("=== capital self-test ===")
    frame = sd.make_ohlcv(n=300, seed=5, start_price=1200.0)
    flow = _fake_flow()
    print("合成资金流：%s，列=%s" % (flow.shape, list(flow.columns)))
    print("派生滚动和示例（末行）：5日 %.2f 亿 / 10日 %.2f 亿 / 20日 %.2f 亿"
          % (flow["main_net_5d"].iloc[-1] / 1e8, flow["main_net_10d"].iloc[-1] / 1e8,
             flow["main_net_20d"].iloc[-1] / 1e8))
    holders, margin, lhb, top10 = _fake_holders(), _fake_margin(), _fake_lhb(), _fake_top10()

    # ---- monkeypatch：把联网 fetch 替换成合成数据（证明因子逻辑可离线验证） -- #
    global fetch_holders, fetch_margin, fetch_lhb, fetch_northbound, \
        fetch_top10_float_holders, fetch_snapshot
    _orig = (fetch_holders, fetch_margin, fetch_lhb, fetch_northbound,
             fetch_top10_float_holders, fetch_snapshot)
    fetch_holders = lambda symbol, force=False: holders
    fetch_margin = lambda symbol, days=60, force=False: margin
    fetch_lhb = lambda symbol, force=False: lhb
    fetch_northbound = lambda symbol, force=False: pd.DataFrame()
    fetch_top10_float_holders = lambda symbol, force=False: top10
    fetch_snapshot = lambda symbol, force=False: {"turnover": 0.28, "volume_ratio": 1.18,
                                                  "float_market_cap": 1.5788e12,
                                                  "market_cap": 1.5788e12, "price": 1263.0}
    try:
        cd = control_degree("600519", frame=frame, holders=holders)
        print("控盘度：score=%.3f level=%s 换手均值=%.2f%% 户数环比=%.1f%% 十大占比=%.1f%%"
              % (cd["score"], cd["level"], cd["turnover_avg"],
                 cd["holder_num_change_pct"], cd["top10_ratio"]))
        print("   ", cd["detail"])
        prof = main_force_profile(flow, frame=frame)
        print("主力行为：%s（置信度 %.2f）20日净额 %.2f 亿，20日涨跌 %+.2f%%"
              % (prof["behavior"], prof["confidence"], prof["net_20d"] / 1e8,
                 prof["price_20d_chg"]))
        print("   ", prof["detail"])

        fs = capital_factors("600519", flow=flow, holders=holders, margin=margin,
                             lhb=lhb, north=None, top10=top10,
                             snapshot={"fund_holding_change": 3.5}, frame=frame)
        ids = sorted({f.id for f in fs})
        print("因子数量：%d" % len(fs))
        print("因子 id 清单（%d）：%s" % (len(ids), ids))
        assert "capital_main_net" in ids and "main_force_behavior" in ids
        assert "control_degree" in ids
        for f in fs:
            print("   %-28s %-4s 强度 %.2f 可信度 %.2f | %s"
                  % (f.id, {1: "看涨", -1: "看跌", 0: "中性"}[f.direction],
                     f.strength, f.reliability, (f.detail or "")[:60]))

        # ---- 数据全空用例（仍不联网：所有 fetch 换成返回空对象） ---------- #
        fetch_holders = lambda *a, **k: pd.DataFrame()
        fetch_margin = lambda *a, **k: pd.DataFrame()
        fetch_lhb = lambda *a, **k: pd.DataFrame()
        fetch_northbound = lambda *a, **k: pd.DataFrame()
        fetch_top10_float_holders = lambda *a, **k: pd.DataFrame()
        fetch_snapshot = lambda *a, **k: {}
        empty = capital_factors("600519")
        print("空数据因子数量：%d，id：%s" % (len(empty), sorted({f.id for f in empty})))
        for f in empty:
            print("   [空] %-28s 方向=%s 可信度 %.2f" %
                  (f.id, f.direction, f.reliability))
            assert f.reliability <= 0.5, "空数据时可信度必须下调"
        assert main_force_profile(None)["behavior"] == "观望"
        cd0 = control_degree("600519")
        print("空数据控盘度：level=%s detail=%s" % (cd0["level"], cd0["detail"][:60]))
        assert cd0["level"] in ("数据不足", "分散", "中度控盘", "高度控盘")
        assert cd0["score"] == 0.0
    finally:
        (fetch_holders, fetch_margin, fetch_lhb, fetch_northbound,
         fetch_top10_float_holders, fetch_snapshot) = _orig

    print("capital self-test OK")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
