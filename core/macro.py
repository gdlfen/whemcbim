# -*- coding: utf-8 -*-
"""
宏观经济周期与行业环境模块（Macro）。

职责
----
把**宏观水位（GDP/PMI/CPI/PPI/M2/社融/利率/汇率）、大盘市场状态（均线系统+波动率+回撤）、
产业政策方向、上游原材料成本**压缩成 :class:`core.contracts.Factor`，
为**长周期预测**提供"市场水位"与"行业景气"判断。

产出因子 id
-----------
``macro_cycle``、``interest_rate``、``inflation``、``liquidity``、``market_regime``、
``industry_policy``、``regulation_risk``、``commodity_cost``、``supply_demand``、``fx_impact``。
其中 ``market_regime`` 与 ``industry_policy`` 覆盖 ``("d20","m1","q1","h1","y1")``。

数据源与实测结论（本机 2026-10 实测）
------------------------------------
============  ============================================================  ==============
目标           调用                                                          排序
============  ============================================================  ==============
CPI 同比       ``ak.macro_china_cpi()`` → ``全国-同比增长``                 最新在最前
PPI 同比       ``ak.macro_china_ppi()`` → ``当月同比增长``                   最新在最前
PMI           ``ak.macro_china_pmi()`` → ``制造业-指数``                    最新在最前
GDP 同比       ``ak.macro_china_gdp()`` → ``国内生产总值-同比增长``          最新在最前
LPR           ``ak.macro_china_lpr()`` → ``TRADE_DATE/LPR1Y/LPR5Y``         最新在最前
M2            ``ak.macro_china_money_supply()`` → ``货币和准货币(M2)-同比增长``  最新在最前
社融           ``ak.macro_china_shrzgm()`` → ``月份/社会融资规模增量``        **升序**，同比自算
SHIBOR 3M     ``ak.rate_interbank(market="上海银行同业拆借市场",
              symbol="Shibor人民币", indicator="3月")`` → ``利率``            升序
指数日线        ``ak.stock_zh_index_daily(symbol="sh000001")``              8738 行
汇率          新浪 ``hq.sinajs.cn/list=fx_susdcny``（GBK + Referer）       逗号分隔，
              **索引 8=最新价、1=昨收**
商品（国际）    新浪 ``list=hf_CL,hf_CAD,hf_GC``（原油/伦铜/纽约黄金）        **索引 0=最新价、
              3=开、4=高、5=低、7=昨收、13=名称**
商品（国内）    ``ak.futures_main_sina("RB0"/"LC0"/"I0")``（螺纹钢/碳酸锂/铁矿石）
              → ``日期/开盘价/最高价/最低价/收盘价``；备选新浪 ``nf_RB0``
VIX          新浪 ``gb_$vix`` 返回空串、yfinance 被限流 → **用 MACRO_DEFAULTS 兜底并记录**
============  ============================================================  ==============

所有来源**各自独立 try**，失败即用 ``config.MACRO_DEFAULTS`` 兜底，并记录
``source`` / ``fallback_fields``；全部函数优雅降级，绝不抛异常。宏观数据缓存 12 小时。
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re

import numpy as np
import pandas as pd

from . import config
from .contracts import (Category, DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Factor,
                        clamp, now_cn, safe_float)
from .utils import (cache_key, cached, get_logger, http_get, retry_call, tanh_scale)

log = get_logger("stockai.macro")

__all__ = ["fetch_macro", "fetch_industry_policy", "fetch_commodity_prices",
           "macro_factors", "market_regime", "industry_cycle",
           "fetch_index_daily", "last_notes", "MACRO_TTL", "COMMODITY_MAP"]

#: 宏观数据缓存有效期（12 小时）
MACRO_TTL = 12 * 3600

_SINA_HQ = "https://hq.sinajs.cn/list=%s"
_SINA_HDR = {"Referer": "https://finance.sina.com.cn"}
_EM_ANN = ("https://np-anotice-stock.eastmoney.com/api/security/ann?sr=-1&page_size=30"
           "&page_index=1&ann_type=A&client_source=web&stock_list=%s")

#: 商品映射：中文名 → (新浪国际代码, akshare 国内主连代码, 新浪国内代码)
COMMODITY_MAP = {
    "原油": ("hf_CL", None, None),
    "铜": ("hf_CAD", None, None),
    "黄金": ("hf_GC", None, None),
    "螺纹钢": (None, "RB0", "nf_RB0"),
    "碳酸锂": (None, "LC0", "nf_LC0"),
    "铁矿石": (None, "I0", "nf_I0"),
}
#: 英文键别名（便于其它模块/报告取用）
COMMODITY_ALIAS = {"原油": "crude_oil", "铜": "copper", "黄金": "gold",
                   "螺纹钢": "rebar", "碳酸锂": "lithium_carbonate", "铁矿石": "iron_ore"}

#: 上游成本敏感行业（商品涨价 = 成本压力）
UPSTREAM_COST_INDUSTRIES = ("钢铁", "有色", "化工", "建材", "电力", "航空", "交运", "造纸",
                            "建筑", "机械", "汽车", "电池", "新能源", "家电")
#: 上游资源行业（商品涨价 = 直接利好）
UPSTREAM_RESOURCE_INDUSTRIES = ("煤炭", "石油", "有色", "钢铁", "锂", "稀土", "矿业", "农业")

#: 政策关键词分组（用于 subsidy / regulation_risk 判定）
SUBSIDY_KEYWORDS = ("补贴", "减税", "税收优惠", "专项债", "以旧换新", "试点", "国产替代",
                    "重大项目", "采购", "中标", "扩容", "降准", "降息")
REGULATION_KEYWORDS = ("反倾销", "制裁", "禁令", "限产", "停产", "处罚", "立案", "调查",
                       "问询", "内卷", "产能过剩", "召回", "退市", "反垄断")

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
    """日线列名容错；非 DataFrame 返回 None。"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    ren = {"日期": "date", "Date": "date", "DATE": "date", "trade_date": "date",
           "开盘价": "open", "最高价": "high", "最低价": "low", "收盘价": "close",
           "成交量": "volume", "开盘": "open", "最高": "high", "最低": "low",
           "收盘": "close", "Close": "close", "Volume": "volume"}
    return df.rename(columns={k: v for k, v in ren.items() if k in df.columns}).copy()


def _neutral(fid: str, name: str, detail: str, reliability: float = 0.2, value=None,
             theory: str = "美林时钟", horizon=None) -> Factor:
    return Factor(fid, name, Category.MACRO, DIR_NEUTRAL, 0.0,
                  clamp(reliability, 0.05, 0.5), value=value, detail=detail,
                  theory=theory, horizon=horizon)


def _last(df: pd.DataFrame, col: str, default=float("nan")) -> float:
    """取某列最后一个有效值（自动兼容升降序）。"""
    try:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.iloc[-1]) if len(s) else default
    except Exception:  # noqa: BLE001
        return default


def _first(df: pd.DataFrame, col: str, default=float("nan")) -> float:
    """取某列第一个有效值（akshare 宏观多数接口最新值在最前）。"""
    try:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.iloc[0]) if len(s) else default
    except Exception:  # noqa: BLE001
        return default


def _ak(fn_name: str, *args, **kwargs):
    """惰性调用 akshare 函数（失败返回 None，绝不抛异常）。"""
    try:
        import akshare as ak
        fn = getattr(ak, fn_name, None)
        if fn is None:
            _note("akshare 无接口 %s" % fn_name)
            return None
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        _note("akshare.%s 调用失败：%r" % (fn_name, exc))
        return None


# --------------------------------------------------------------------------- #
# 1. 宏观数据
# --------------------------------------------------------------------------- #
def _fetch_macro_raw() -> dict:
    """逐个来源取宏观数据，返回 ``(values, sources, fallbacks)``。"""
    vals: dict = {}
    src: dict = {}
    fb: list = []

    def _try(field: str, label: str, fn) -> None:
        try:
            v = fn()
            if v is None or (isinstance(v, float) and math.isnan(v)):
                raise ValueError("空值")
            vals[field] = float(v)
            src[field] = label
        except Exception as exc:  # noqa: BLE001
            fb.append(field)
            _note("%s(%s) 取数失败，使用默认值 %.3f：%r"
                  % (field, label, config.MACRO_DEFAULTS[field], exc))

    # ---- GDP 同比 -------------------------------------------------------- #
    def _gdp():
        df = _ak("macro_china_gdp")
        return _first(df, "国内生产总值-同比增长") if isinstance(df, pd.DataFrame) else None
    _try("gdp_yoy", "akshare.macro_china_gdp", _gdp)

    # ---- CPI 同比（注意不是 macro_china_cpi_monthly，那个是月率） -------- #
    def _cpi():
        df = _ak("macro_china_cpi")
        return _first(df, "全国-同比增长") if isinstance(df, pd.DataFrame) else None
    _try("cpi_yoy", "akshare.macro_china_cpi", _cpi)

    # ---- PPI 同比 -------------------------------------------------------- #
    def _ppi():
        df = _ak("macro_china_ppi")
        return _first(df, "当月同比增长") if isinstance(df, pd.DataFrame) else None
    _try("ppi_yoy", "akshare.macro_china_ppi", _ppi)

    # ---- PMI ------------------------------------------------------------- #
    def _pmi():
        df = _ak("macro_china_pmi")
        return _first(df, "制造业-指数") if isinstance(df, pd.DataFrame) else None
    _try("pmi", "akshare.macro_china_pmi", _pmi)

    # ---- LPR ------------------------------------------------------------- #
    def _lpr():
        df = _ak("macro_china_lpr")
        if not isinstance(df, pd.DataFrame):
            return None
        vals["lpr_1y"] = _last(df, "LPR1Y")
        vals["lpr_5y"] = _last(df, "LPR5Y")
        src["lpr_1y"] = src["lpr_5y"] = "akshare.macro_china_lpr"
        return None
    try:
        _lpr()
        for k in ("lpr_1y", "lpr_5y"):
            if k not in vals or math.isnan(_num(vals.get(k))):
                raise ValueError("空值")
    except Exception as exc:  # noqa: BLE001
        for k in ("lpr_1y", "lpr_5y"):
            fb.append(k)
            vals.pop(k, None)
        _note("LPR 取数失败，使用默认值：%r" % exc)

    # ---- 3 个月 SHIBOR --------------------------------------------------- #
    def _shibor():
        df = _ak("rate_interbank", market="上海银行同业拆借市场",
                 symbol="Shibor人民币", indicator="3月")
        return _last(df, "利率") if isinstance(df, pd.DataFrame) else None
    _try("shibor_3m", "akshare.rate_interbank(SHIBOR 3M)", _shibor)

    # ---- M2 同比 --------------------------------------------------------- #
    def _m2():
        df = _ak("macro_china_money_supply")
        return _first(df, "货币和准货币(M2)-同比增长") if isinstance(df, pd.DataFrame) else None
    _try("m2_yoy", "akshare.macro_china_money_supply", _m2)

    # ---- 社融同比（接口为升序的月度增量 → 用滚动 12 个月合计同比，口径更稳） -- #
    def _sf():
        df = _ak("macro_china_shrzgm")
        if not isinstance(df, pd.DataFrame) or "月份" not in df.columns:
            return None
        d = df.copy()
        d["_m"] = d["月份"].astype(str).str.strip()
        d["_v"] = pd.to_numeric(d["社会融资规模增量"], errors="coerce")
        d = d.dropna(subset=["_v"]).sort_values("_m")
        if len(d) < 25:
            return None
        ttm = float(d["_v"].tail(12).sum())              # 最近 12 个月合计（滚动）
        prev = float(d["_v"].iloc[-24:-12].sum())        # 前 12 个月合计
        if prev == 0:
            return None
        return (ttm / abs(prev) - 1.0) * 100.0
    _try("social_financing_yoy", "akshare.macro_china_shrzgm(自算同比)", _sf)

    # ---- 美元人民币汇率（新浪，索引 8=最新价） --------------------------- #
    def _fx():
        txt = http_get(_SINA_HQ % "fx_susdcny", encoding="gbk", headers=_SINA_HDR)
        if not txt or '="' not in txt:
            return None
        body = txt.split('="', 1)[1].rsplit('"', 1)[0]
        p = body.split(",")
        for idx in (8, 1, 2):       # 实测 8=最新价，1=昨收
            v = _num(p[idx]) if len(p) > idx else float("nan")
            if not math.isnan(v) and v > 0:
                return v
        return None
    _try("usd_cny", "sina.fx_susdcny", _fx)

    # ---- VIX（新浪 gb_$vix 实测返回空；yfinance 本机被限流） ------------- #
    def _vix():
        try:
            import yfinance as yf
            try:
                yf.set_tz_cache_location(r"E:\deepseek desktop\个股预测程序\.cache\yf")
            except Exception:  # noqa: BLE001
                pass
            h = yf.Ticker("^VIX").history(period="5d")
            if isinstance(h, pd.DataFrame) and not h.empty:
                return float(h["Close"].dropna().iloc[-1])
        except Exception as exc:  # noqa: BLE001
            log.debug("yfinance VIX 失败: %r", exc)
        txt = http_get(_SINA_HQ % "gb_$vix", encoding="gbk", headers=_SINA_HDR)
        if txt and '="' in txt:
            body = txt.split('="', 1)[1].rsplit('"', 1)[0]
            v = _num(body.split(",")[0]) if body else float("nan")
            if not math.isnan(v) and v > 0:
                return v
        return None
    _try("vix", "yfinance ^VIX / sina gb_$vix", _vix)
    return {"values": vals, "sources": src, "fallbacks": fb}


def fetch_macro(force=False) -> dict:
    """宏观经济数据（键与 ``config.MACRO_DEFAULTS`` 对齐）。

    Returns
    -------
    dict
        ``gdp_yoy, cpi_yoy, ppi_yoy, pmi, lpr_1y, lpr_5y, shibor_3m, m2_yoy,
        social_financing_yoy, usd_cny, vix`` + ``source``（字段→来源标签）、
        ``fallback_fields``（用默认值兜底的字段）、``as_of``、``detail``。

    每个来源独立 try；失败字段用 ``config.MACRO_DEFAULTS`` 兜底并记录，缓存 12 小时。
    """
    key = cache_key("macro", now_cn().strftime("%Y%m%d"))

    def _p() -> dict:
        out = dict(config.MACRO_DEFAULTS)
        try:
            raw = _fetch_macro_raw()
        except Exception as exc:  # noqa: BLE001
            _note("宏观取数整体异常，全部使用默认值：%r" % exc)
            raw = {"values": {}, "sources": {}, "fallbacks": list(config.MACRO_DEFAULTS)}
        out.update({k: v for k, v in raw["values"].items() if k in config.MACRO_DEFAULTS})
        fbs = sorted(set(list(raw["fallbacks"]) +
                         [k for k in config.MACRO_DEFAULTS
                          if k not in raw["values"]]))
        out["source"] = {k: raw["sources"].get(k, "default")
                         for k in config.MACRO_DEFAULTS}
        out["fallback_fields"] = fbs
        out["as_of"] = now_cn().strftime("%Y-%m-%d %H:%M")
        out["detail"] = ("宏观数据：%d/%d 项取得实时值%s"
                         % (len(config.MACRO_DEFAULTS) - len(fbs),
                            len(config.MACRO_DEFAULTS),
                            "；兜底字段：" + "、".join(fbs) if fbs else ""))
        _note(out["detail"])
        return out

    try:
        if force:
            return _p()
        out = cached(key, MACRO_TTL, _p)
        return out if isinstance(out, dict) else dict(config.MACRO_DEFAULTS)
    except Exception as exc:  # noqa: BLE001
        _note("fetch_macro 异常降级：%r" % exc)
        out = dict(config.MACRO_DEFAULTS)
        out.update({"source": {}, "fallback_fields": list(config.MACRO_DEFAULTS),
                    "detail": "宏观取数异常，全部使用默认值"})
        return out


# --------------------------------------------------------------------------- #
# 2. 指数与市场状态
# --------------------------------------------------------------------------- #
def _producer_index_daily() -> pd.DataFrame:
    """上证指数日线（akshare，实测 8738 行）；失败回退新浪指数行情。"""
    df = _ak("stock_zh_index_daily", symbol=config.RUNTIME["index_symbol"])
    if isinstance(df, pd.DataFrame) and not df.empty:
        out = _ensure_frame(df)
        if out is not None and "close" in out.columns:
            _note("指数日线来源：akshare.stock_zh_index_daily(sh000001)，%d 根" % len(out))
            return out
    try:       # 备选：新浪指数日线（与个股同接口）
        txt = http_get("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                       "CN_MarketData.getKLineData?symbol=sh000001&scale=240&ma=no"
                       "&datalen=1200", headers=_SINA_HDR)
        data = json.loads(txt) if txt else []
        if data:
            out = pd.DataFrame([{"date": pd.Timestamp(r["day"]), "open": _num(r["open"]),
                                 "high": _num(r["high"]), "low": _num(r["low"]),
                                 "close": _num(r["close"]), "volume": _num(r["volume"])}
                                for r in data])
            _note("指数日线降级来源：新浪指数日线（%d 根）" % len(out))
            return out
    except Exception as exc:  # noqa: BLE001
        _note("新浪指数日线失败：%r" % exc)
    _note("指数日线全部来源失败")
    return pd.DataFrame()


def fetch_index_daily(force=False) -> pd.DataFrame:
    """基准指数（上证指数）日线；失败返回空 DataFrame。"""
    try:
        if force:
            return _producer_index_daily()
        out = cached(cache_key("index", config.RUNTIME["index_symbol"]),
                     config.RUNTIME["cache_ttl_seconds"], _producer_index_daily)
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        _note("fetch_index_daily 异常降级：%r" % exc)
        return pd.DataFrame()


def market_regime(index_data, force=False) -> dict:
    """大盘状态判定（**长周期预测的"市场水位"**）。

    判定逻辑（均线系统 + 波动率 + 回撤）：

    * ``ma_state``：MA5>MA20>MA60 → 多头；MA5<MA20<MA60 → 空头；否则纠缠。
    * ``trend_score`` = 0.45×均线排列分 + 0.35×动量分 + 0.20×回撤分（∈[-1,1]）：
      均线排列分 = (2×满足条数/条数 − 1)；动量分 = tanh(近 20 日收益 / 8%)；
      回撤分 = clamp(距 250 日高点回撤% / 25, −1, 0)。
    * ``regime``：多头 + 站上 MA250 + trend_score>0.3 → **牛市**；
      空头 + 跌破 MA250 + trend_score<−0.3 → **熊市**；
      深度回撤（<−15%）且 MA5 上穿 MA20（近 10 根内）→ **反弹中**；
      高位（回撤 > −8%）且 trend_score<−0.15 → **调整中**；其余 **震荡市**。

    Returns
    -------
    dict
        ``{"regime","ma_state","annual_vol_pct","index_pe_percentile","above_ma250",
        "drawdown_from_high_pct","trend_score","detail"}``；数据不足时
        ``regime="未知"``、``trend_score=0``。
    """
    out = {"regime": "未知", "ma_state": "未知", "annual_vol_pct": float("nan"),
           "index_pe_percentile": None, "above_ma250": False,
           "drawdown_from_high_pct": float("nan"), "trend_score": 0.0,
           "detail": "指数数据不足，无法判定市场状态"}
    try:
        df = _ensure_frame(index_data)
        if df is None:
            df = fetch_index_daily(force=force)
        df = _ensure_frame(df)
        if df is None or "close" not in df.columns:
            return out
        d = df.copy()
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            d = d.dropna(subset=["date"]).sort_values("date")
        c = pd.to_numeric(d["close"], errors="coerce").dropna().reset_index(drop=True)
        if len(c) < 60:
            out["detail"] = "指数样本仅 %d 根，不足以判定市场状态" % len(c)
            return out
        ma = {n: float(c.rolling(n, min_periods=max(2, n // 2)).mean().iloc[-1])
              for n in (5, 20, 60, 120, 250)}
        conds = [ma[5] > ma[20], ma[20] > ma[60],
                 ma[60] > ma[120] if not math.isnan(ma[120]) else ma[60] > ma[20],
                 float(c.iloc[-1]) > ma[250] if not math.isnan(ma[250])
                 else float(c.iloc[-1]) > ma[60]]
        valid = [x for x in conds[:2] + [conds[3]] if x is not None]
        ma_frac = sum(1 for x in valid if x) / max(1, len(valid))
        ma_score = 2 * ma_frac - 1
        if ma[5] > ma[20] > ma[60]:
            ma_state = "多头"
        elif ma[5] < ma[20] < ma[60]:
            ma_state = "空头"
        else:
            ma_state = "纠缠"
        mom = (float(c.iloc[-1]) / float(c.iloc[-21]) - 1.0) * 100.0 if len(c) > 21 else 0.0
        mom_score = tanh_scale(mom, 8.0)
        win = c.tail(250)
        dd = (float(c.iloc[-1]) / float(win.max()) - 1.0) * 100.0 if len(win) else 0.0
        dd_score = clamp(dd / 25.0, -1.0, 0.0)
        trend = clamp(0.45 * ma_score + 0.35 * mom_score + 0.20 * dd_score, -1, 1)
        # 年化波动率（60 日）
        ret = c.pct_change().dropna().tail(60)
        vol = float(ret.std(ddof=1) * math.sqrt(config.RUNTIME["trading_days_per_year"]) * 100) \
            if len(ret) > 5 else float("nan")
        above250 = float(c.iloc[-1]) > ma[250] if not math.isnan(ma[250]) else \
            float(c.iloc[-1]) > ma[60]
        # 近 10 根内 MA5 是否上穿 MA20（用于识别反弹）
        cross_up = False
        try:
            s5 = c.rolling(5).mean()
            s20 = c.rolling(20).mean()
            diff = (s5 - s20).dropna().tail(12)
            cross_up = len(diff) >= 2 and diff.iloc[-1] > 0 >= diff.min()
        except Exception:  # noqa: BLE001
            cross_up = False
        regime = "震荡市"
        if ma_state == "多头" and above250 and trend > 0.3:
            regime = "牛市"
        elif ma_state == "空头" and not above250 and trend < -0.3:
            regime = "熊市"
        elif dd < -15 and cross_up:
            regime = "反弹中"
        elif dd > -8 and trend < -0.15:
            regime = "调整中"
        out.update({
            "regime": regime, "ma_state": ma_state,
            "annual_vol_pct": vol, "above_ma250": bool(above250),
            "drawdown_from_high_pct": dd, "trend_score": round(float(trend), 3),
            "ma": {("MA%d" % k): round(v, 2) for k, v in ma.items()},
            "index_close": float(c.iloc[-1]),
            "last_date": str(d["date"].iloc[-1].date()) if "date" in d.columns
            and len(d) else "",
        })
        out["detail"] = ("大盘 %s：均线%s、收盘 %.0f（MA20 %.0f / MA60 %.0f / MA250 %.0f），"
                         "距 250 日高点 %.1f%%，年化波动 %.1f%%，趋势分 %.2f"
                         % (regime, ma_state, float(c.iloc[-1]), ma[20], ma[60], ma[250],
                            dd, vol, trend))
        _note("市场状态：%s" % out["detail"])
    except Exception as exc:  # noqa: BLE001
        out["detail"] = "市场状态判定异常，已降级：%r" % exc
        _note(out["detail"])
    return out


# --------------------------------------------------------------------------- #
# 3. 产业政策
# --------------------------------------------------------------------------- #
def _fetch_titles(symbol, limit: int = 30) -> list:
    """取最近公告标题（东财 ``np-anotice``，实测可用）。"""
    url = _EM_ANN % re.sub(r"\D", "", str(symbol or ""))[:6]
    payload = retry_call(lambda: http_get(url, json_body=True), attempts=3, delay=0.8,
                         default=None, label="ann_policy")
    rows = ((payload or {}).get("data") or {}).get("list") or []
    out = []
    for r in rows[:limit]:
        out.append({"title": str(r.get("title") or ""),
                    "date": str(r.get("notice_date") or "")[:10],
                    "type": ((r.get("columns") or [{}])[0].get("column_name")
                             if r.get("columns") else "")})
    return out


def fetch_industry_policy(symbol, name="", force=False) -> dict:
    """产业政策方向打分（对最近新闻/公告标题做政策关键词匹配）。

    Returns
    -------
    dict
        ``{"policy_score":-1~1,"matched":[{"keyword","title","direction"}],
        "themes":[...],"subsidy":bool,"regulation_risk":bool,"detail"}``。

    说明：东财搜索接口在本机返回与关键词无关的固定内容，因此实际标题来源是
    **东财公告接口**（``np-anotice``）；新闻标题可由调用方通过 ``name`` 之外的渠道补充
    （见 :func:`core.newsflow.fetch_news`）。
    """
    key = cache_key("policy", re.sub(r"\D", "", str(symbol or ""))[:6], name)

    def _p() -> dict:
        out = {"policy_score": 0.0, "matched": [], "themes": [], "subsidy": False,
               "regulation_risk": False, "detail": "", "n_titles": 0}
        titles = _fetch_titles(symbol)
        if name:
            titles = titles + [{"title": str(name), "date": "", "type": "名称"}]
        out["n_titles"] = len(titles)
        good = config.POLICY_KEYWORDS.get("利好", [])
        bad = config.POLICY_KEYWORDS.get("利空", [])
        score = 0.0
        matched = []
        for it in titles:
            t = str(it.get("title") or "")
            for kw in good:
                if kw in t:
                    score += 1.0
                    matched.append({"keyword": kw, "title": t, "direction": 1})
            for kw in bad:
                if kw in t:
                    score -= 1.0
                    matched.append({"keyword": kw, "title": t, "direction": -1})
        themes = [th for th in config.HOT_THEMES
                  if th and (th in str(name) or any(th in str(x.get("title") or "")
                                                    for x in titles))]
        blob = " ".join(str(x.get("title") or "") for x in titles)
        out.update({
            "policy_score": round(float(math.tanh(score / 3.0)), 3),
            "matched": matched[:20], "themes": themes,
            "subsidy": any(k in blob for k in SUBSIDY_KEYWORDS),
            "regulation_risk": any(k in blob for k in REGULATION_KEYWORDS),
            "detail": ("分析 %d 条公告/名称标题，命中利好关键词 %d 次、利空 %d 次；"
                       "题材：%s；政策支持 %s；监管风险 %s"
                       % (len(titles), sum(1 for m in matched if m["direction"] > 0),
                          sum(1 for m in matched if m["direction"] < 0),
                          "、".join(themes) if themes else "无",
                          "是" if out["subsidy"] else "否",
                          "是" if out["regulation_risk"] else "否")),
        })
        if not titles:
            out["detail"] = "未取到任何公告标题（东财公告接口失败），政策打分取中性 0"
        _note("产业政策：%s" % out["detail"])
        return out

    try:
        if force:
            return _p()
        out = cached(key, MACRO_TTL, _p)
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001
        _note("fetch_industry_policy 异常降级：%r" % exc)
        return {"policy_score": 0.0, "matched": [], "themes": [], "subsidy": False,
                "regulation_risk": False, "detail": "取数异常，已降级"}


# --------------------------------------------------------------------------- #
# 4. 商品价格
# --------------------------------------------------------------------------- #
def _sina_hf(code: str) -> dict:
    """新浪国际商品（hf_ 系列）。

    实测格式：``var hq_str_hf_CL="91.716,,91.640,91.670,92.170,90.010,04:59:58,91.490,
    91.290,0,3,1,2026-10-10,纽约原油,0";``
    → **索引 0=最新价、3=开盘、4=最高、5=最低、6=时间、7=昨收、12=日期、13=名称**。
    """
    txt = http_get(_SINA_HQ % code, encoding="gbk", headers=_SINA_HDR)
    if not txt or '="' not in txt:
        return {}
    p = txt.split('="', 1)[1].rsplit('"', 1)[0].split(",")
    if len(p) < 8:
        return {}
    price, prev = _num(p[0]), _num(p[7])
    return {"price": price, "prev_close": prev, "open": _num(p[3]), "high": _num(p[4]),
            "low": _num(p[5]), "date": p[12] if len(p) > 12 else "",
            "name": p[13] if len(p) > 13 else code,
            "pct_chg": ((price / prev - 1.0) * 100.0) if prev else float("nan"),
            "source": "sina." + code}


def _sina_nf(code: str) -> dict:
    """新浪国内期货（nf_ 系列）。实测索引：1=时间、2=开、3=高、4=低、6=买价、7=卖价、
    8=最新价、10=昨收/昨结、13=持仓、14=成交、16=品种、17=日期。"""
    txt = http_get(_SINA_HQ % code, encoding="gbk", headers=_SINA_HDR)
    if not txt or '="' not in txt:
        return {}
    p = txt.split('="', 1)[1].rsplit('"', 1)[0].split(",")
    if len(p) < 11:
        return {}
    price, prev = _num(p[8]), _num(p[10])
    return {"price": price, "prev_close": prev, "open": _num(p[2]), "high": _num(p[3]),
            "low": _num(p[4]), "date": p[17] if len(p) > 17 else "",
            "name": p[16] if len(p) > 16 else code,
            "pct_chg": ((price / prev - 1.0) * 100.0) if prev else float("nan"),
            "source": "sina." + code}


def _ak_futures_main(code: str) -> dict:
    """akshare 国内主连日线（实测 `RB0` 可用）→ 取最后两根收盘价。"""
    end = now_cn().date()
    start = end - dt.timedelta(days=20)
    df = _ak("futures_main_sina", symbol=code,
             start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    d = _ensure_frame(df)
    if d is None or "close" not in d.columns:
        return {}
    c = pd.to_numeric(d["close"], errors="coerce").dropna()
    if not len(c):
        return {}
    price = float(c.iloc[-1])
    prev = float(c.iloc[-2]) if len(c) > 1 else float("nan")
    return {"price": price, "prev_close": prev,
            "pct_chg": ((price / prev - 1.0) * 100.0) if prev and prev > 0 else float("nan"),
            "date": str(d["date"].iloc[-1])[:10] if "date" in d.columns else "",
            "name": code, "source": "akshare.futures_main_sina." + code}


def fetch_commodity_prices(force=False) -> dict:
    """上游原材料价格（原油/铜/黄金/螺纹钢/碳酸锂/铁矿石）。

    Returns
    -------
    dict
        中文键 → ``{"price","prev_close","pct_chg","open","high","low","date","name",
        "source"}``；另含 ``"alias"``（英文键别名映射）、``"missing"``（未取到的品种）、
        ``"detail"``。失败品种直接跳过（**不报错**）。
    """
    key = cache_key("commodity", now_cn().strftime("%Y%m%d%H"))

    def _p() -> dict:
        out: dict = {}
        missing = []
        for cname, (hf, ak_code, nf) in COMMODITY_MAP.items():
            got = {}
            for fn in (lambda: _sina_hf(hf) if hf else {},
                       lambda: _ak_futures_main(ak_code) if ak_code else {},
                       lambda: _sina_nf(nf) if nf else {}):
                try:
                    got = fn() or {}
                except Exception as exc:  # noqa: BLE001
                    log.debug("商品 %s 取数失败: %r", cname, exc)
                    got = {}
                if got and not math.isnan(_num(got.get("price"))):
                    break
            if got and not math.isnan(_num(got.get("price"))):
                out[cname] = got
            else:
                missing.append(cname)
        out["alias"] = {COMMODITY_ALIAS[k]: v for k, v in out.items()
                        if k in COMMODITY_ALIAS and isinstance(v, dict)}
        out["missing"] = missing
        out["detail"] = ("上游商品 %d/%d 个品种取到价格%s"
                         % (len(COMMODITY_MAP) - len(missing), len(COMMODITY_MAP),
                            "；未取到：" + "、".join(missing) if missing else ""))
        _note(out["detail"])
        return out

    try:
        if force:
            return _p()
        out = cached(key, MACRO_TTL, _p)
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001
        _note("fetch_commodity_prices 异常降级：%r" % exc)
        return {}


# --------------------------------------------------------------------------- #
# 5. 行业周期
# --------------------------------------------------------------------------- #
def industry_cycle(symbol, name="", force=False) -> dict:
    """行业周期阶段（复苏/繁荣/衰退/萧条）。

    推断方法（**权重写进注释**）：

    * PMI 水位（40%）：``clamp((PMI−50)/3, −1, 1)``——制造业景气度。
    * 个股相对大盘强度（30%）：近 60 日超额收益 ``clamp(rs/10, −1, 1)``
      （价格数据取不到时该项剔除、按剩余权重归一化）。
    * 商品价格动量（30%）：螺纹钢/碳酸锂/铜等均价动量（周期行业成本/需求代理）。

    阶段划分：score ≥ 0.35 繁荣；0 ~ 0.35 复苏；−0.35 ~ 0 衰退；< −0.35 萧条。

    Returns
    -------
    dict
        ``{"stage","score","detail"}``。
    """
    out = {"stage": "未知", "score": 0.0, "detail": ""}
    try:
        macro = fetch_macro(force=force)
        pmi = _num(macro.get("pmi"), config.MACRO_DEFAULTS["pmi"])
        pmi_score = clamp((pmi - 50.0) / 3.0, -1, 1)
        parts = ["PMI %.1f（%s）" % (pmi, "扩张" if pmi >= 50 else "收缩")]
        comps = [(pmi_score, 0.40)]
        # ---- 相对强度 ----------------------------------------------------- #
        rs = float("nan")
        try:
            from .fundamental import fetch_price_series          # 惰性导入，避免循环依赖
            px = fetch_price_series(symbol)
            idx = fetch_index_daily(force=force)
            if isinstance(px, pd.Series) and len(px) > 60 and isinstance(idx, pd.DataFrame) \
                    and not idx.empty:
                c = pd.to_numeric(idx["close"], errors="coerce").dropna()
                if len(c) > 60:
                    n = min(len(px), len(c), 61)
                    rs = ((float(px.iloc[-1]) / float(px.iloc[-n]) - 1.0)
                          - (float(c.iloc[-1]) / float(c.iloc[-n]) - 1.0)) * 100.0
        except Exception as exc:  # noqa: BLE001
            log.debug("行业周期相对强度失败: %r", exc)
        if not math.isnan(rs):
            comps.append((clamp(rs / 10.0, -1, 1), 0.30))
            parts.append("近 60 日相对大盘 %+.1f%%" % rs)
        # ---- 商品动量 ----------------------------------------------------- #
        cm = fetch_commodity_prices(force=force)
        pcts = [_num(v.get("pct_chg")) for k, v in (cm or {}).items()
                if isinstance(v, dict) and k in ("螺纹钢", "碳酸锂", "铜", "原油")]
        pcts = [x for x in pcts if not math.isnan(x)]
        if pcts:
            cm_score = clamp(float(np.mean(pcts)) / 3.0, -1, 1)
            comps.append((cm_score, 0.30))
            parts.append("上游商品日均涨跌 %+.2f%%" % float(np.mean(pcts)))
        tot = sum(w for _, w in comps)
        score = sum(v * w for v, w in comps) / tot if tot else 0.0
        stage = ("繁荣" if score >= 0.35 else "复苏" if score >= 0 else "衰退"
                 if score >= -0.35 else "萧条")
        out.update({"stage": stage, "score": round(float(score), 3),
                    "detail": "行业周期判定为「%s」（综合分 %.2f）：%s"
                              % (stage, score, "；".join(parts))})
        if len(comps) < 3:
            out["detail"] += "；数据不全（缺少相对强度或商品价格），可信度下调"
    except Exception as exc:  # noqa: BLE001
        out["detail"] = "行业周期判定异常，已降级：%r" % exc
        _note(out["detail"])
    return out


# --------------------------------------------------------------------------- #
# 6. 因子合成
# --------------------------------------------------------------------------- #
def macro_factors(symbol, macro=None, policy=None, commodity=None, index_data=None,
                  frame=None) -> list:
    """产出宏观因子列表。

    Parameters
    ----------
    macro : dict | None
        :func:`fetch_macro` 的产出；``None`` 时用 ``config.MACRO_DEFAULTS`` 兜底（可信度 ≤0.3）。
    policy : dict | None
        :func:`fetch_industry_policy` 的产出。
    commodity : dict | None
        :func:`fetch_commodity_prices` 的产出。
    index_data : pd.DataFrame | None
        指数日线；提供时计算 :func:`market_regime`（**不联网**），否则该因子取中性。
    frame : pd.DataFrame | None
        个股日线（用于判断个股相对强度，可空）。

    Returns
    -------
    list[Factor]
        id 见模块 docstring；全空输入也不抛异常。
    """
    fs: list = []
    try:
        mk = dict(config.MACRO_DEFAULTS)
        if isinstance(macro, dict) and macro:
            mk.update({k: _num(macro.get(k), config.MACRO_DEFAULTS[k])
                       for k in config.MACRO_DEFAULTS})
            real = [k for k in config.MACRO_DEFAULTS
                    if (macro.get("source") or {}).get(k, "default") != "default"]
            rel_base = 0.7 if len(real) >= 8 else (0.5 if len(real) >= 4 else 0.3)
            has_macro = len(real) > 0
        else:
            rel_base = 0.25
            has_macro = False
            _note("macro_factors：未提供宏观数据，使用 config.MACRO_DEFAULTS 兜底（低可信度）")
        pol = policy if isinstance(policy, dict) else {}
        com = commodity if isinstance(commodity, dict) else {}
        long_h = ("q1", "h1", "y1")
        regime_h = ("d20", "m1", "q1", "h1", "y1")

        # ---- 1) 宏观周期（GDP/PMI） --------------------------------------- #
        pmi, gdp = mk["pmi"], mk["gdp_yoy"]
        score = clamp((pmi - 50.0) / 3.0, -1, 1) * 0.6 + clamp(gdp / 5.0 - 1.0, -1, 1) * 0.4
        fs.append(Factor("macro_cycle", "宏观周期", Category.MACRO,
                         DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.1 else 0),
                         clamp(abs(score), 0, 1), clamp(rel_base, 0.15, 0.8),
                         value={"PMI": pmi, "GDP同比": gdp},
                         detail="PMI %.1f（%s）、GDP 同比 %.1f%%，经济%s"
                                % (pmi, "荣枯线上" if pmi >= 50 else "荣枯线下", gdp,
                                   "扩张" if score > 0.1 else
                                   ("收缩" if score < -0.1 else "平稳")),
                         theory="美林时钟 / 经济周期", horizon=long_h))

        # ---- 2) 利率（下行=利好估值） ------------------------------------- #
        lpr, shibor = mk["lpr_1y"], mk["shibor_3m"]
        d_lpr = lpr - config.MACRO_DEFAULTS["lpr_1y"]
        d_sh = shibor - config.MACRO_DEFAULTS["shibor_3m"]
        score = -clamp(d_lpr / 0.5, -1, 1) * 0.6 - clamp(d_sh / 0.5, -1, 1) * 0.4
        fs.append(Factor("interest_rate", "利率环境", Category.MACRO,
                         DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.1 else 0),
                         clamp(abs(score), 0, 1), clamp(rel_base * 0.9, 0.15, 0.8),
                         value={"LPR1Y": lpr, "SHIBOR3M": shibor},
                         detail="LPR(1年) %.2f%%、SHIBOR(3月) %.2f%%，利率%s（相对基准 "
                                "%+.2f/+%.2f 个百分点）"
                                % (lpr, shibor,
                                   "下行利好估值" if score > 0.1 else
                                   ("上行压制估值" if score < -0.1 else "基本平稳"),
                                   d_lpr, d_sh),
                         theory="DCF 估值 / 流动性", horizon=long_h))

        # ---- 3) 通胀（CPI/PPI 组合） -------------------------------------- #
        cpi, ppi = mk["cpi_yoy"], mk["ppi_yoy"]
        s_cpi = -abs(cpi - 1.5) / 3.0 + 0.5          # CPI 1.5% 附近最舒服
        s_ppi = clamp(ppi / 5.0, -1, 1)              # PPI 通缩→上游承压
        score = clamp(s_cpi, -1, 1) * 0.5 + s_ppi * 0.5
        detail = ("CPI %.1f%%、PPI %.1f%%，%s"
                  % (cpi, ppi,
                     "温和通胀、需求健康" if (0.5 <= cpi <= 3.0 and ppi > 0) else
                     ("通缩压力（PPI 为负），中下游成本改善但需求偏弱" if ppi < 0 else
                      "通胀偏高，货币政策空间受限")))
        fs.append(Factor("inflation", "通胀环境", Category.MACRO,
                         DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.1 else 0),
                         clamp(abs(score), 0, 1), clamp(rel_base * 0.85, 0.15, 0.75),
                         value={"CPI": cpi, "PPI": ppi}, detail=detail,
                         theory="通胀-盈利传导", horizon=long_h))

        # ---- 4) 流动性（M2/社融/SHIBOR） ---------------------------------- #
        m2, sf = mk["m2_yoy"], mk["social_financing_yoy"]
        s_m2 = clamp((m2 - config.MACRO_DEFAULTS["m2_yoy"]) / 2.0, -1, 1)
        s_sf = clamp((sf - config.MACRO_DEFAULTS["social_financing_yoy"]) / 5.0, -1, 1)
        s_sh = -clamp(d_sh / 0.5, -1, 1)
        score = s_m2 * 0.35 + s_sf * 0.35 + s_sh * 0.30
        fs.append(Factor("liquidity", "流动性", Category.MACRO,
                         DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.1 else 0),
                         clamp(abs(score), 0, 1), clamp(rel_base * 0.9, 0.15, 0.8),
                         value={"M2同比": m2, "社融同比": sf, "SHIBOR3M": shibor},
                         detail="M2 同比 %.1f%%、社融同比 %.1f%%、拆借利率 %.2f%%，"
                                "资金面%s" % (m2, sf, shibor,
                                          "宽松" if score > 0.1 else
                                          ("偏紧" if score < -0.1 else "中性")),
                         theory="流动性驱动 / 剩余流动性", horizon=long_h))

        # ---- 5) 市场状态（大盘水位，长周期权重高） ------------------------ #
        rg = market_regime(index_data, force=False) if index_data is not None else \
            {"regime": "未知", "trend_score": 0.0, "detail": "未提供指数数据（本因子不联网）"}
        tscore = _num(rg.get("trend_score"), 0.0)
        if rg.get("regime") == "未知" or not index_data is not None:
            fs.append(_neutral("market_regime", "大盘市场状态",
                               rg.get("detail") or "缺少指数数据", 0.2,
                               value=rg.get("regime"), theory="道氏理论 / 均线系统",
                               horizon=regime_h))
        else:
            fs.append(Factor("market_regime", "大盘市场状态", Category.MACRO,
                             DIR_BULL if tscore > 0.12 else (DIR_BEAR if tscore < -0.12 else 0),
                             clamp(abs(tscore), 0, 1), 0.8,
                             value={"regime": rg.get("regime"), "trend_score": tscore},
                             detail=rg.get("detail", ""),
                             theory="道氏理论 / 均线系统 + 波动率", horizon=regime_h))

        # ---- 6) 产业政策 -------------------------------------------------- #
        pscore = _num(pol.get("policy_score"), 0.0)
        if not pol:
            fs.append(_neutral("industry_policy", "产业政策", "未提供政策数据", 0.2,
                               theory="政策驱动", horizon=regime_h))
        else:
            fs.append(Factor("industry_policy", "产业政策", Category.MACRO,
                             DIR_BULL if pscore > 0.1 else (DIR_BEAR if pscore < -0.1 else 0),
                             clamp(abs(pscore), 0, 1), 0.5,
                             value={"评分": pscore, "题材": pol.get("themes")},
                             detail=pol.get("detail", ""),
                             theory="政策驱动 / 产业周期", horizon=regime_h))

        # ---- 7) 监管风险 -------------------------------------------------- #
        risk = bool(pol.get("regulation_risk"))
        fs.append(Factor("regulation_risk", "监管风险", Category.MACRO,
                         DIR_BEAR if risk else DIR_NEUTRAL, 0.55 if risk else 0.0,
                         0.5 if pol else 0.2, value=risk,
                         detail="检测到监管/限产类关键词（反倾销、制裁、限产、处罚等），"
                                "存在政策扰动风险" if risk else
                                "未检测到监管风险关键词",
                         theory="政策风险溢价", horizon=("d20", "m1", "q1")))

        # ---- 8) 原材料成本 ------------------------------------------------ #
        cm_pcts = {k: _num(v.get("pct_chg")) for k, v in (com or {}).items()
                   if isinstance(v, dict) and "pct_chg" in v}
        cm_pcts = {k: v for k, v in cm_pcts.items() if not math.isnan(v)}
        if not cm_pcts:
            fs.append(_neutral("commodity_cost", "原材料成本", "未取到商品价格数据", 0.2,
                               theory="成本传导", horizon=("q1", "h1", "y1")))
        else:
            avg = float(np.mean(list(cm_pcts.values())))
            # 无法可靠判断个股处于上游还是下游 → 用中性方向 + 低可信度，detail 说明
            fs.append(Factor("commodity_cost", "原材料成本", Category.MACRO, DIR_NEUTRAL,
                             0.0, 0.3,
                             value=cm_pcts,
                             detail="上游商品价格当日平均 %+.2f%%（%s）；个股处于产业链"
                                    "上游则利好、下游则成本承压，需结合行业判断"
                                    % (avg, "、".join("%s%+.2f%%" % (k, v)
                                                      for k, v in cm_pcts.items())),
                             theory="成本传导 / 产业链利润分配", horizon=("q1", "h1", "y1")))

        # ---- 9) 供需景气 -------------------------------------------------- #
        pmi_score = clamp((pmi - 50.0) / 3.0, -1, 1)
        cm_score = clamp(float(np.mean(list(cm_pcts.values()))) / 3.0, -1, 1) \
            if cm_pcts else 0.0
        score = pmi_score * 0.7 + cm_score * 0.3
        fs.append(Factor("supply_demand", "供需景气", Category.MACRO,
                         DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.1 else 0),
                         clamp(abs(score), 0, 1), clamp(rel_base * 0.8, 0.15, 0.7),
                         value={"PMI": pmi, "商品动量": cm_score},
                         detail="制造业 PMI %.1f、上游商品动量 %.2f，供需%s"
                                % (pmi, cm_score,
                                   "改善" if score > 0.1 else
                                   ("走弱" if score < -0.1 else "平稳")),
                         theory="供需缺口 / 库存周期", horizon=long_h))

        # ---- 10) 汇率影响 ------------------------------------------------- #
        fx = mk["usd_cny"]
        fx_chg = (fx / config.MACRO_DEFAULTS["usd_cny"] - 1.0) * 100.0
        fs.append(Factor("fx_impact", "汇率影响", Category.MACRO, DIR_NEUTRAL,
                         0.0, 0.3,
                         value={"USDCNY": fx, "相对基准变化%": fx_chg},
                         detail="美元兑人民币 %.4f（相对配置基准 %+.2f%%）；人民币贬值"
                                "利好出口型、利空进口型/外币负债型公司，本股类型未知故不"
                                "给方向" % (fx, fx_chg),
                         theory="汇率传导", horizon=("q1", "h1", "y1")))
        if not has_macro:
            for f in fs:
                f.reliability = clamp(f.reliability * 0.8, 0.05, 0.5)
    except Exception as exc:  # noqa: BLE001
        _note("macro_factors 内部异常（已降级）：%r" % exc)
        fs.append(_neutral("macro_cycle", "宏观周期", "宏观因子生成异常，已降级", 0.1))
    return fs


# --------------------------------------------------------------------------- #
# 自测
# --------------------------------------------------------------------------- #
def _fake_macro() -> dict:
    out = dict(config.MACRO_DEFAULTS)
    out.update({"gdp_yoy": 5.2, "cpi_yoy": 0.8, "ppi_yoy": -1.2, "pmi": 50.6,
                "lpr_1y": 2.9, "lpr_5y": 3.4, "shibor_3m": 1.43, "m2_yoy": 8.4,
                "social_financing_yoy": 9.1, "usd_cny": 6.69, "vix": 15.2,
                "source": {k: "selftest" for k in config.MACRO_DEFAULTS},
                "fallback_fields": [], "as_of": "2026-10-11 10:00",
                "detail": "selftest"})
    return out


def _self_test() -> None:
    """离线自测：合成指数 + 合成宏观/政策/商品，**不发网络请求**。

    为了让 ``industry_cycle`` / ``market_regime`` 完全不联网，这里把模块内的
    取数函数临时替换成合成实现（monkeypatch），结束后恢复。
    """
    import core.selftest_data as sd
    import core.fundamental as _fd

    print("=== macro self-test ===")
    index = sd.make_index(n=800, seed=3)
    stock = sd.make_ohlcv(n=800, seed=9)
    macro, policy = _fake_macro(), {
        "policy_score": 0.42, "matched": [{"keyword": "补贴", "title": "获得政府补贴",
                                           "direction": 1}],
        "themes": ["白酒"], "subsidy": True, "regulation_risk": False,
        "detail": "selftest：命中利好 1 次"}
    commodity = {"原油": {"price": 91.7, "prev_close": 91.5, "pct_chg": 0.22},
                 "铜": {"price": 14553.0, "prev_close": 14540.0, "pct_chg": 0.09},
                 "螺纹钢": {"price": 3080.0, "prev_close": 3061.0, "pct_chg": 0.62},
                 "碳酸锂": {"price": 117000.0, "prev_close": 118740.0, "pct_chg": -1.47},
                 "missing": ["黄金"], "detail": "selftest"}

    global fetch_macro, fetch_commodity_prices, fetch_index_daily
    _orig = (fetch_macro, fetch_commodity_prices, fetch_index_daily,
             _fd.fetch_price_series)
    fetch_macro = lambda force=False: macro
    fetch_commodity_prices = lambda force=False: commodity
    fetch_index_daily = lambda force=False: pd.DataFrame()
    _fd.fetch_price_series = lambda symbol, bars=1300, force=False: pd.Series(
        stock["close"].values, index=pd.to_datetime(stock["date"]))
    try:
        rg = market_regime(index)
        print("市场状态：%s（均线 %s，趋势分 %.2f，年化波动 %.1f%%，回撤 %.1f%%）"
              % (rg["regime"], rg["ma_state"], rg["trend_score"],
                 _num(rg["annual_vol_pct"]), _num(rg["drawdown_from_high_pct"])))
        print("   ", rg["detail"])
        cyc = industry_cycle("600519", name="贵州茅台")
        print("行业周期：%s（score %.2f）" % (cyc["stage"], cyc["score"]))
        print("   ", cyc["detail"])

        fs = macro_factors("600519", macro=macro, policy=policy, commodity=commodity,
                           index_data=index, frame=stock)
        ids = sorted({f.id for f in fs})
        print("因子数量：%d" % len(fs))
        print("因子 id 清单（%d）：%s" % (len(ids), ids))
        for f in fs:
            print("   %-20s %-4s 强度 %.2f 可信度 %.2f | %s"
                  % (f.id, {1: "看涨", -1: "看跌", 0: "中性"}[f.direction], f.strength,
                     f.reliability, (f.detail or "")[:60]))
        assert "market_regime" in ids and "interest_rate" in ids and "fx_impact" in ids
        assert all(f.category == Category.MACRO for f in fs)

        # ---- 全空用例（不联网：不传 index_data，market_regime 保持中性） --- #
        empty = macro_factors("600519")
        print("空数据因子数量：%d，id：%s" % (len(empty), sorted({f.id for f in empty})))
        assert all(f.reliability <= 0.5 for f in empty)
        rg0 = market_regime(pd.DataFrame())
        print("空指数数据市场状态：%s trend_score=%s detail=%s"
              % (rg0["regime"], rg0["trend_score"], rg0["detail"][:60]))
        assert rg0["regime"] in ("未知", "震荡市", "牛市", "熊市", "反弹中", "调整中")
    finally:
        (fetch_macro, fetch_commodity_prices, fetch_index_daily,
         _fd.fetch_price_series) = _orig

    print("macro self-test OK")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
