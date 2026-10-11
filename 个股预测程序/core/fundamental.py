# -*- coding: utf-8 -*-
"""
基本面分析模块（Fundamental）。

职责
----
把公司的**财务质量、成长性、现金流、负债安全、估值水位与事件催化剂**压缩成
:class:`core.contracts.Factor` 列表，供多周期概率预测引擎使用。

产出因子 id
-----------
成长：``fundamental_growth``（营收/净利同比 + 季度环比加速度 + 连续改善）、
      ``growth_divergence``（单季营收与净利增速背离）
质量：``roe_quality``、``margin_trend``、``profit_quality``（扣非占比）
现金流：``cashflow_health``（经营现金流/净利润、同比、每股经营现金流）
安全：``leverage_risk``（资产负债率、流动比率、有息负债压力）
估值：``valuation_percentile``（PE/PB 历史分位，低分位看涨）、``peg_level``、
      ``relative_valuation``（对行业中位数）
事件：``earnings_catalyst``、``turnaround``、``restructure_event``
持续性：``receivable_risk``（应收/存货周转恶化、研发费用率）、``loss_warning``

数据源与降级链
--------------
* 财务摘要：``akshare.stock_financial_abstract``（主，80 项指标 × 100+ 报告期）
  → 利润表补全 ``akshare.stock_profit_sheet_by_report_em``（营业利润/利润总额/研发费用）
  → 东财 ``RPT_LICO_FN_CPD`` → 空 ``DataFrame``。
* 实时估值：腾讯 ``qt.gtimg.cn``（GBK，字段索引见 :data:`TENCENT_FIELD`，**本机实测固化**）
  → 派生 PE/PB 历史分位（盈利不变假设，见 :func:`valuation_percentile`）。
* 价格序列：新浪日线（1300 根 ≈ 5 年） → 腾讯前复权日线。
* 同行对比：东财行业板块（行业代码取 ``RPT_LICO_FN_CPD.BOARD_CODE``，实测 600519 → BK1277 白酒Ⅱ）
  → 板块成分 ``push2.eastmoney.com/api/qt/clist/get``（本机该域名**间歇性** RemoteDisconnected，
  统一用 :func:`core.utils.retry_call` 重试 3 次）。

所有联网函数**优雅降级**：失败时返回空 ``DataFrame`` / ``{}`` 并写 ``log.debug``，绝不抛异常。
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
                    percentile_rank, retry_call)

log = get_logger("stockai.fundamental")

__all__ = [
    "fetch_financials", "fetch_valuation", "fetch_industry_peers",
    "fundamental_factors", "growth_quality_score", "valuation_percentile",
    "earnings_catalyst", "to_ak_symbol",
    "FIN_COLUMNS", "TENCENT_FIELD", "last_notes",
]

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
#: 规范财务表列（缺失列一律留 NaN，绝不臆造）
FIN_COLUMNS = [
    "报告期", "营业收入", "营收同比", "净利润", "净利同比", "扣非净利润", "扣非净利同比",
    "ROE", "摊薄ROE", "毛利率", "净利率", "资产负债率", "经营现金流", "经营现金流同比",
    "每股收益", "每股净资产", "每股经营现金流", "存货周转率", "应收账款周转率",
    "流动比率", "速动比率", "营业利润", "利润总额", "研发费用",
]
#: 派生列（在 FIN_COLUMNS 之外附加）
FIN_DERIVED = ["报告期类型", "单季营收", "单季净利", "营收环比", "净利环比", "环比增速",
               "研发费用率", "扣非占比", "经营现金流净利比"]

#: akshare ``stock_financial_abstract`` 的「指标」→ 规范列名（**精确匹配**，避免子串误配）
ABSTRACT_MAP = {
    "营业总收入": "营业收入",
    "营业总收入增长率": "营收同比",
    "归母净利润": "净利润",              # 与「归属母公司净利润增长率」口径一致
    "归属母公司净利润增长率": "净利同比",
    "扣非净利润": "扣非净利润",
    "净资产收益率(ROE)": "ROE",
    "摊薄净资产收益率": "摊薄ROE",
    "毛利率": "毛利率",
    "销售净利率": "净利率",
    "资产负债率": "资产负债率",
    "经营现金流量净额": "经营现金流",
    "基本每股收益": "每股收益",
    "每股净资产": "每股净资产",
    "每股经营现金流": "每股经营现金流",
    "存货周转率": "存货周转率",
    "应收账款周转率": "应收账款周转率",
    "流动比率": "流动比率",
    "速动比率": "速动比率",
}

#: akshare 利润表（东财口径）→ 规范列名，用于补全营业利润/利润总额/研发费用
INCOME_MAP = {
    "TOTAL_OPERATE_INCOME": "营业收入",
    "OPERATE_PROFIT": "营业利润",
    "TOTAL_PROFIT": "利润总额",
    "RESEARCH_EXPENSE": "研发费用",
    "PARENT_NETPROFIT": "净利润",
    "DEDUCT_PARENT_NETPROFIT": "扣非净利润",
    "DEDUCT_PARENT_NETPROFIT_YOY": "扣非净利同比",
}

#: 东财 ``RPT_LICO_FN_CPD``（财务快报）→ 规范列名；该接口同时给出所属行业板块
EM_FIN_MAP = {
    "TOTAL_OPERATE_INCOME": "营业收入",
    "YSTZ": "营收同比",
    "PARENT_NETPROFIT": "净利润",
    "SJLTZ": "净利同比",
    "WEIGHTAVG_ROE": "ROE",
    "BPS": "每股净资产",
    "MGJYXJJE": "每股经营现金流",
    "XSMLL": "毛利率",
    "BASIC_EPS": "每股收益",
    "YSHZ": "营收环比",
    "SJLHZ": "净利环比",
}

#: 腾讯实时快照字段索引 —— **本机实测固化**（``https://qt.gtimg.cn/q=sh600519``，GBK，``~`` 分隔）
#: 实测样本（2026-10-09 贵州茅台）：[3]=1263.00 现价, [4]=1255.79 昨收, [5]=1255.42 今开,
#: [6]=35111 成交量(手), [30]=20261009161459 时间戳, [31]=7.21 涨跌额, [32]=0.57 涨跌幅%,
#: [33]=1282.00 最高, [34]=1255.42 最低, [37]=445322 成交额(万元), [38]=0.28 换手率%,
#: [39]=19.39 PE(TTM), [43]=2.12 振幅%, [44]=15788.53 流通市值(亿), [45]=15788.53 总市值(亿),
#: [46]=6.28 PB, [47]=1381.37 涨停价, [48]=1130.21 跌停价, [49]=1.18 量比, [50]=78 委差,
#: [51]=1268.34 均价, [52]=17.73 PE(动), [53]=19.18 PE(静), [57]=445322.3189 成交额(万元),
#: [64]=4.12 股息率%(用 601398/000858/601318 三只股票交叉验证：4.12/3.77/7.31/5.25 与
#:   近 12 个月每股分红÷股价一致), [72]=1250081601 总股本, [73]=1250081601 流通股本,
#: [67]/[68] 疑似 52 周最高/最低（仅作参考，未用于因子强弱）。
TENCENT_FIELD = {
    "name": 1, "code": 2, "price": 3, "prev_close": 4, "open": 5, "volume": 6,
    "time": 30, "change": 31, "pct_chg": 32, "high": 33, "low": 34, "amount_wan": 37,
    "turnover": 38, "pe_ttm": 39, "amplitude": 43, "float_market_cap_yi": 44,
    "market_cap_yi": 45, "pb": 46, "limit_up": 47, "limit_down": 48,
    "volume_ratio": 49, "avg_price": 51, "pe_dynamic": 52, "pe_static": 53,
    "dividend_yield": 64, "total_shares": 72, "float_shares": 73,
    "week52_high": 67, "week52_low": 68,
}

_TENCENT_URL = "https://qt.gtimg.cn/q=%s"
_EM_DATACENTER = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_EM_CLIST = "https://push2.eastmoney.com/api/qt/clist/get"
_EM_SLIST = "https://push2.eastmoney.com/api/qt/slist/get"
_SINA_KLINE = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               "CN_MarketData.getKLineData?symbol=%s&scale=240&ma=no&datalen=%d")
_TENCENT_KLINE = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=%s,day,,,%d,qfq")

#: 报告期新鲜度阈值（天）：超过则认为财报过时，因子可信度打折
STALE_DAYS = 180
VERY_STALE_DAYS = 365

#: 用于「周期股」判断的行业关键词（周期股 PE 高不代表贵，PEG 不可靠）
CYCLICAL_KEYWORDS = ("钢铁", "煤炭", "有色", "化工", "水泥", "航运", "港口", "石油",
                     "化纤", "造纸", "猪肉", "养殖", "证券", "房地产", "航空", "玻璃",
                     "锂", "光伏", "半导体", "面板", "工程机械")

#: 最近一次取数过程的降级说明（供报告/自测查看）
_LAST_NOTES: list[str] = []


def last_notes() -> list:
    """返回最近一次取数/因子生成的降级说明（中文），便于报告展示与自测断言。"""
    return list(_LAST_NOTES)


def _note(msg: str) -> None:
    if msg and msg not in _LAST_NOTES:
        _LAST_NOTES.append(msg)
    log.debug(msg)


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _ensure_frame(df):
    """列名容错：把大写/中文列名映射成小写英文标准列；不是 DataFrame 时返回 None。"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    ren = {"日期": "date", "时间": "date", "Date": "date", "DATE": "date", "trade_date": "date",
           "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
           "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_chg", "换手率": "turnover",
           "Open": "open", "Close": "close", "High": "high", "Low": "low", "Volume": "volume"}
    return df.rename(columns={k: v for k, v in ren.items() if k in df.columns}).copy()


def _num(x, default=float("nan")) -> float:
    """安全转 float（兼容 numpy 标量 / 带 % 或 , 的字符串）。"""
    return safe_float(x, default)


def _last_valid(series, default=float("nan")) -> float:
    """取序列中最后一个非 NaN 值。"""
    try:
        arr = pd.Series(series).astype(float).dropna()
        return float(arr.iloc[-1]) if len(arr) else default
    except Exception:  # noqa: BLE001
        return default


def _freshness_mult(frame) -> float:
    """按最近报告期距今天数调整可信度：>180 天 ×0.7，>365 天 ×0.5。"""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return 1.0
    try:
        if "报告期" not in frame.columns:
            return 1.0
        d = pd.to_datetime(frame["报告期"], errors="coerce").max()
        if pd.isna(d):
            return 1.0
        days = (now_cn().date() - d.date()).days
        if days > VERY_STALE_DAYS:
            return 0.5
        if days > STALE_DAYS:
            return 0.7
        return 1.0
    except Exception:  # noqa: BLE001
        return 1.0


def _reliability(frame, base: float, n_obs: int = 0) -> float:
    """按 DEVELOPER_BRIEF 第 5 节标尺 + 报告期新鲜度给出 reliability。"""
    r = base
    if n_obs and n_obs < 4:
        r = min(r, 0.55)
    elif n_obs and n_obs < 8:
        r = min(r, 0.75)
    return clamp(r * _freshness_mult(frame), 0.05, 0.97)


def _neutral(fid: str, name: str, detail: str, reliability: float = 0.2,
             value=None, theory: str = "", horizon=None) -> Factor:
    """数据缺失时的中性因子（direction=0/strength=0），保证因子 id 集合稳定。"""
    return Factor(fid, name, Category.FUND, DIR_NEUTRAL, 0.0, clamp(reliability, 0.05, 0.5),
                  value=value, detail=detail, theory=theory, horizon=horizon)


def to_ak_symbol(symbol: str) -> str:
    """转成 akshare 常用代码：A 股 6 位数字（``600519``）；其它市场返回规范化 code。"""
    try:
        info = normalize_symbol(symbol)
        return info["code"]
    except Exception:  # noqa: BLE001
        return re.sub(r"\D", "", str(symbol or ""))[:6]


def _secid(symbol: str) -> str:
    try:
        return normalize_symbol(symbol)["secid"]
    except Exception:  # noqa: BLE001
        return "1.600519"


def _prefix_symbol(symbol: str) -> str:
    """``sh600519`` 形式（新浪/腾讯用）。"""
    try:
        info = normalize_symbol(symbol)
        return info["display"]
    except Exception:  # noqa: BLE001
        return "sh600519"


def _secucode(symbol: str) -> str:
    """``600519.SH`` 形式（东财 F10 用）。"""
    try:
        info = normalize_symbol(symbol)
        return info["code"] + ("." + info["prefix"].upper() if info["is_cn"] else "")
    except Exception:  # noqa: BLE001
        return "600519.SH"


def _norm_period(col) -> str:
    """把 ``20260630`` / ``2026-06-30`` / ``2026Q2`` 统一成 ``YYYY-MM-DD``，失败返回 ""。"""
    s = str(col).strip()
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", s)
    if m:
        return "%s-%s-%s" % m.groups()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2}).*", s)
    if m:
        return "%s-%s-%s" % m.groups()
    m = re.fullmatch(r"(\d{4})\s*[Qq](\d)", s)
    if m:
        md = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}.get(m.group(2))
        return "%s-%s" % (m.group(1), md) if md else ""
    return ""


def _period_type(period: str) -> str:
    """报告期类型：Q1 / H1 / Q3 / 年报。"""
    md = str(period)[5:10]
    return {"03-31": "Q1", "06-30": "H1", "09-30": "Q3", "12-31": "年报"}.get(md, "其它")


# --------------------------------------------------------------------------- #
# 1. 财务摘要
# --------------------------------------------------------------------------- #
def _abstract_to_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """把 akshare ``stock_financial_abstract`` 的「指标 × 报告期」宽表转置成规范长表。

    原始结构：第一列 ``选项``、第二列 ``指标``，其余列名为 ``20260630`` 这样的报告期。
    注意：报告期为**累计值**（Q1/H1/Q3/年报均为年初至今），单季值与环比由
    :func:`_finalize_financials` 做差得到。
    """
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()
    cols = list(raw.columns)
    if len(cols) < 3:
        return pd.DataFrame()
    periods = [c for c in cols[2:] if re.fullmatch(r"\d{8}", str(c).strip())]
    if not periods:
        return pd.DataFrame()
    idx_col = "指标" if "指标" in raw.columns else cols[1]
    buckets: dict[str, dict] = {}
    for _, row in raw.iterrows():
        target = ABSTRACT_MAP.get(str(row.get(idx_col, "")).strip())
        if not target:
            continue
        for p in periods:
            v = _num(row.get(p))
            if math.isnan(v):
                continue
            slot = buckets.setdefault(str(p), {})
            if target not in slot or (isinstance(slot[target], float)
                                      and math.isnan(slot[target])):
                slot[target] = v
    if not buckets:
        return pd.DataFrame()
    rows = []
    for p, vals in buckets.items():
        rec = {"报告期": _norm_period(p)}
        rec.update(vals)
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out


def _income_to_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """把 akshare 东财利润表转成规范列子集（补全营业利润/利润总额/研发费用）。"""
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()
    if "REPORT_DATE" not in raw.columns:
        return pd.DataFrame()
    keep = {"REPORT_DATE": "报告期"}
    for src, dst in INCOME_MAP.items():
        if src in raw.columns:
            keep[src] = dst
    sub = raw[list(keep)].rename(columns=keep).copy()
    sub["报告期"] = sub["报告期"].map(lambda x: str(x)[:10])
    for c in sub.columns:
        if c != "报告期":
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    return sub


def _em_fin_to_frame(rows: list) -> pd.DataFrame:
    """东财 ``RPT_LICO_FN_CPD`` 记录 → 规范列子集（备选源）。"""
    if not rows:
        return pd.DataFrame()
    out = []
    for r in rows:
        rec = {"报告期": str(r.get("REPORTDATE") or "")[:10]}
        for src, dst in EM_FIN_MAP.items():
            rec[dst] = _num(r.get(src))
        rec["_行业"] = r.get("BOARD_NAME")
        rec["_行业代码"] = r.get("BOARD_CODE")
        out.append(rec)
    return pd.DataFrame(out)


def _finalize_financials(df: pd.DataFrame) -> pd.DataFrame:
    """统一收尾：补列、解析报告期、排序、算单季/环比/派生比率。"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "报告期" not in df.columns:
        return pd.DataFrame()
    df["报告期"] = df["报告期"].map(_norm_period)
    df = df[df["报告期"] != ""]
    if df.empty:
        return pd.DataFrame()
    for c in FIN_COLUMNS:
        if c not in df.columns and c != "报告期":
            df[c] = np.nan
    for c in FIN_COLUMNS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(subset=["报告期"], keep="first")
    df = df.sort_values("报告期").reset_index(drop=True)
    df["报告期类型"] = df["报告期"].map(_period_type)

    # --- 同比补算（源数据缺失时用同报告期去年值自算） --------------------- #
    def _yoy(col: str, out_col: str) -> None:
        if col not in df.columns:
            return
        series = df[col].astype(float)
        prev = []
        lut = {p: v for p, v in zip(df["报告期"], series)}
        for p in df["报告期"]:
            last_year = "%d%s" % (int(p[:4]) - 1, p[4:])
            prev.append(lut.get(last_year, np.nan))
        prev = pd.Series(prev, index=df.index)
        calc = (series / prev.abs() - 1.0) * 100.0
        calc = calc.where(prev.notna() & (prev.abs() > 1e-9))
        df[out_col] = df[out_col].astype(float).where(df[out_col].notna(), calc)

    _yoy("营业收入", "营收同比")
    _yoy("净利润", "净利同比")
    _yoy("扣非净利润", "扣非净利同比")
    _yoy("经营现金流", "经营现金流同比")

    # --- 单季值（报告期为累计口径，做差还原单季） ------------------------- #
    single_rev, single_np = [], []
    lut_rev = dict(zip(df["报告期"], df["营业收入"]))
    lut_np = dict(zip(df["报告期"], df["净利润"]))
    prev_map = {"H1": "03-31", "Q3": "06-30", "年报": "09-30"}
    for p, ptype in zip(df["报告期"], df["报告期类型"]):
        year = int(p[:4])
        base = prev_map.get(ptype)
        pr = "%d-%s" % (year, base) if base else None
        for col, lut, target in (("营业收入", lut_rev, single_rev),
                                 ("净利润", lut_np, single_np)):
            cur = lut.get(p, np.nan)
            if pr and pr in lut:
                val = cur - lut[pr]
            else:
                val = cur
            target.append(val)
    df["单季营收"] = pd.to_numeric(pd.Series(single_rev, index=df.index), errors="coerce")
    df["单季净利"] = pd.to_numeric(pd.Series(single_np, index=df.index), errors="coerce")

    # --- 环比（单季 vs 上一季度单季） ------------------------------------- #
    for col, out in (("单季营收", "营收环比"), ("单季净利", "净利环比")):
        prev = df[col].shift(1)
        calc = (df[col] / prev.abs() - 1.0) * 100.0
        calc = calc.where(prev.notna() & (prev.abs() > 1e-9))
        if out in df.columns:
            df[out] = pd.to_numeric(df[out], errors="coerce").where(
                pd.to_numeric(df[out], errors="coerce").notna(), calc)
        else:
            df[out] = calc
    # 综合环比增速：优先净利环比，缺则用营收环比（列名按需求保留为「环比增速」）
    df["环比增速"] = df["净利环比"].where(df["净利环比"].notna(), df["营收环比"])

    # --- 派生比率 --------------------------------------------------------- #
    with np.errstate(divide="ignore", invalid="ignore"):
        df["研发费用率"] = np.where(pd.to_numeric(df["营业收入"], errors="coerce").abs() > 0,
                                    pd.to_numeric(df["研发费用"], errors="coerce")
                                    / pd.to_numeric(df["营业收入"], errors="coerce") * 100.0,
                                    np.nan)
        df["扣非占比"] = np.where(pd.to_numeric(df["净利润"], errors="coerce").abs() > 0,
                                  pd.to_numeric(df["扣非净利润"], errors="coerce")
                                  / pd.to_numeric(df["净利润"], errors="coerce") * 100.0,
                                  np.nan)
        df["经营现金流净利比"] = np.where(
            pd.to_numeric(df["净利润"], errors="coerce").abs() > 0,
            pd.to_numeric(df["经营现金流"], errors="coerce")
            / pd.to_numeric(df["净利润"], errors="coerce"), np.nan)

    cols = ["报告期"] + [c for c in FIN_COLUMNS if c != "报告期"] + FIN_DERIVED
    cols = [c for c in cols if c in df.columns]
    for c in ("_行业", "_行业代码"):
        if c in df.columns:
            cols.append(c)
    return df[cols].reset_index(drop=True)


def _producer_financials(symbol: str) -> pd.DataFrame:
    """财务摘要取数（不含缓存）：akshare 摘要 + 利润表补全 → 东财快报 → 空表。"""
    code = to_ak_symbol(symbol)
    frame = pd.DataFrame()
    # --- 主源：akshare 财务摘要 ------------------------------------------- #
    try:
        import akshare as ak
        raw = ak.stock_financial_abstract(symbol=code)
        frame = _abstract_to_frame(raw)
        if not frame.empty:
            _note("财务摘要来源：akshare.stock_financial_abstract（%d 期）" % len(frame))
    except Exception as exc:  # noqa: BLE001
        _note("akshare 财务摘要失败：%r" % exc)
        frame = pd.DataFrame()

    # --- 利润表补全：营业利润/利润总额/研发费用 --------------------------- #
    if not frame.empty:
        try:
            import akshare as ak
            prefix = _prefix_symbol(symbol).upper()
            inc = retry_call(lambda: ak.stock_profit_sheet_by_report_em(symbol=prefix),
                             attempts=2, delay=0.6, default=None,
                             label="profit_sheet_em")
            sub = _income_to_frame(inc)
            if not sub.empty:
                frame = frame.merge(sub, on="报告期", how="outer", suffixes=("", "_inc"))
                for c in list(frame.columns):
                    if c.endswith("_inc"):
                        base = c[:-4]
                        if base in frame.columns:
                            frame[base] = frame[base].where(frame[base].notna(),
                                                            frame[c])
                        else:
                            frame[base] = frame[c]
                        frame = frame.drop(columns=[c])
                _note("利润表补全成功：营业利润/利润总额/研发费用来自东财利润表")
        except Exception as exc:  # noqa: BLE001
            _note("利润表补全失败（营业利润/利润总额/研发费用留 NaN）：%r" % exc)

    # --- 备选源：东财财务快报 --------------------------------------------- #
    if frame.empty:
        try:
            url = (_EM_DATACENTER + "?reportName=RPT_LICO_FN_CPD&columns=ALL"
                   "&filter=(SECUCODE=\"%s\")&pageSize=40"
                   "&sortColumns=REPORTDATE&sortTypes=-1" % _secucode(symbol))
            payload = retry_call(lambda: http_get(url, json_body=True), attempts=3,
                                 delay=0.8, default=None, label="RPT_LICO_FN_CPD")
            rows = ((payload or {}).get("result") or {}).get("data") or []
            frame = _em_fin_to_frame(rows)
            if not frame.empty:
                _note("财务摘要降级来源：东财 RPT_LICO_FN_CPD（%d 期，"
                      "营业利润/利润总额/研发费用缺失）" % len(frame))
        except Exception as exc:  # noqa: BLE001
            _note("东财财务快报失败：%r" % exc)
            frame = pd.DataFrame()

    out = _finalize_financials(frame)
    if out.empty:
        _note("财务数据全部来源失败，返回空表（基本面因子将退化为中性/低可信度）")
    return out


def fetch_financials(symbol, force=False) -> pd.DataFrame:
    """取规范财务摘要表（见 :data:`FIN_COLUMNS` + :data:`FIN_DERIVED`）。

    列：``报告期, 营业收入, 营收同比, 净利润, 净利同比, 扣非净利润, 扣非净利同比, ROE,
    摊薄ROE, 毛利率, 净利率, 资产负债率, 经营现金流, 经营现金流同比, 每股收益, 每股净资产,
    每股经营现金流, 存货周转率, 应收账款周转率, 流动比率, 速动比率, 营业利润, 利润总额,
    研发费用`` + ``报告期类型, 单季营收, 单季净利, 营收环比, 净利环比, 环比增速, 研发费用率,
    扣非占比, 经营现金流净利比``。按 ``报告期`` 升序；缺失列留 NaN。

    失败返回空 ``DataFrame``（不抛异常）。
    """
    code = to_ak_symbol(symbol)
    key = cache_key("fin", code)
    try:
        if force:
            return _producer_financials(symbol)
        out = cached(key, config.RUNTIME["cache_ttl_daily"],
                     lambda: _producer_financials(symbol))
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        _note("fetch_financials 异常降级：%r" % exc)
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# 2. 价格序列与估值
# --------------------------------------------------------------------------- #
def _producer_daily_close(symbol: str, bars: int = 1300) -> pd.Series:
    """日线收盘价序列（升序，索引为 Timestamp）：新浪 → 腾讯前复权 → 空 Series。"""
    psym = _prefix_symbol(symbol)
    # --- 新浪（不复权，最长 1300+ 根） ------------------------------------ #
    try:
        txt = http_get(_SINA_KLINE % (psym, bars),
                       headers={"Referer": "https://finance.sina.com.cn"})
        data = json.loads(txt) if txt else []
        if data:
            s = pd.Series({pd.Timestamp(str(r["day"])): _num(r["close"]) for r in data})
            s = s.dropna().sort_index()
            if len(s) >= 60:
                return s
    except Exception as exc:  # noqa: BLE001
        _note("新浪日线失败：%r" % exc)
    # --- 腾讯前复权 ------------------------------------------------------- #
    try:
        payload = http_get(_TENCENT_KLINE % (psym, max(320, bars)), json_body=True)
        arr = ((payload or {}).get("data") or {}).get(psym, {})
        rows = arr.get("qfqday") or arr.get("day") or []
        if rows:
            s = pd.Series({pd.Timestamp(r[0]): _num(r[2]) for r in rows})  # [date,O,C,H,L,V]
            s = s.dropna().sort_index()
            if len(s):
                _note("价格序列降级来源：腾讯前复权日线（%d 根）" % len(s))
                return s
    except Exception as exc:  # noqa: BLE001
        _note("腾讯日线失败：%r" % exc)
    return pd.Series(dtype=float)


def fetch_price_series(symbol, bars: int = 1300, force=False) -> pd.Series:
    """取日线收盘价序列（升序 ``pd.Series``，索引 Timestamp）；失败返回空 Series。"""
    code = to_ak_symbol(symbol)
    key = cache_key("close", code, bars)
    try:
        if force:
            return _producer_daily_close(symbol, bars)
        out = cached(key, config.RUNTIME["cache_ttl_seconds"],
                     lambda: _producer_daily_close(symbol, bars))
        return out if isinstance(out, pd.Series) else pd.Series(dtype=float)
    except Exception as exc:  # noqa: BLE001
        _note("fetch_price_series 异常降级：%r" % exc)
        return pd.Series(dtype=float)


def _producer_tencent_quote(symbol: str) -> dict:
    """腾讯实时快照 → dict（GBK 解码，字段索引见 :data:`TENCENT_FIELD`）。"""
    psym = _prefix_symbol(symbol)
    txt = http_get(_TENCENT_URL % psym, encoding="gbk")
    if not txt or '="' not in txt:
        return {}
    body = txt.split('="', 1)[1].rsplit('"', 1)[0]
    parts = body.split("~")
    if len(parts) < 60:
        return {}
    out = {}
    for key, idx in TENCENT_FIELD.items():
        if idx >= len(parts):
            continue
        raw = parts[idx]
        out[key] = raw if key in ("name", "code", "time") else _num(raw)
    # 单位换算：市值 亿 → 元
    for k in ("market_cap_yi", "float_market_cap_yi"):
        if k in out:
            out[k.replace("_yi", "")] = out[k] * 1e8
    if "amount_wan" in out:
        out["amount"] = out["amount_wan"] * 1e4
    out["source"] = "tencent"
    return out


def fetch_quote(symbol, force=False) -> dict:
    """腾讯实时快照（含 PE/PB/市值/换手/股息率）；失败返回 ``{}``。"""
    code = to_ak_symbol(symbol)
    key = cache_key("quote", code)
    try:
        if force:
            return _producer_tencent_quote(symbol) or {}
        out = cached(key, config.RUNTIME["cache_ttl_seconds"],
                     lambda: _producer_tencent_quote(symbol))
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001
        _note("腾讯快照失败：%r" % exc)
        return {}


def _eps_ttm(frame) -> float:
    """滚动 12 个月每股收益 = 最近年报 EPS + 最新期 EPS − 去年同期 EPS（累计口径）。"""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return float("nan")
    if "每股收益" not in frame.columns:
        return float("nan")
    df = frame.dropna(subset=["每股收益"]).copy()
    if df.empty:
        return float("nan")
    df = df.sort_values("报告期")
    latest = df.iloc[-1]
    p = str(latest["报告期"])
    if _period_type(p) == "年报":
        return _num(latest["每股收益"])
    ann = df[df["报告期"].astype(str).str.endswith("12-31")
             & (df["报告期"].astype(str).str[:4] == str(int(p[:4]) - 1))]
    prev = df[df["报告期"].astype(str) == "%d%s" % (int(p[:4]) - 1, p[4:])]
    if not ann.empty and not prev.empty:
        val = _num(ann.iloc[-1]["每股收益"]) + _num(latest["每股收益"]) \
            - _num(prev.iloc[-1]["每股收益"])
        if not math.isnan(val):
            return val
    return _num(latest["每股收益"])


def _bps_latest(frame) -> float:
    """最新一期每股净资产。"""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return float("nan")
    if "每股净资产" not in frame.columns:
        return float("nan")
    return _last_valid(frame.sort_values("报告期")["每股净资产"])


def _ttm_sum(frame, col: str) -> float:
    """滚动 12 个月合计（累计口径）：最近年报 + 最新期 − 去年同期。"""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return float("nan")
    if col not in frame.columns:
        return float("nan")
    df = frame.dropna(subset=[col]).sort_values("报告期")
    if df.empty:
        return float("nan")
    latest = df.iloc[-1]
    p = str(latest["报告期"])
    if _period_type(p) == "年报":
        return _num(latest[col])
    ann = df[df["报告期"].astype(str).str.endswith("12-31")
             & (df["报告期"].astype(str).str[:4] == str(int(p[:4]) - 1))]
    prev = df[df["报告期"].astype(str) == "%d%s" % (int(p[:4]) - 1, p[4:])]
    if not ann.empty and not prev.empty:
        return _num(ann.iloc[-1][col]) + _num(latest[col]) - _num(prev.iloc[-1][col])
    return _num(latest[col])


def valuation_percentile(price_series, frame, method="pe") -> dict:
    """用「盈利不变假设」估算估值历史分位。

    原理：无法直接拿到历史 PE/PB 序列时，用 **日线收盘价 ÷ 最近一期每股收益(TTM)**
    （``method="pe"``）或 **收盘价 ÷ 每股净资产**（``method="pb"``）构造近似估值序列，
    再用 :func:`core.utils.percentile_rank` 求当前值在窗口内的分位（0~1）。
    窗口取最近 5 年（不足则全部）。该近似忽略盈利变化，**周期股/业绩剧烈变化时偏差大**。

    Returns
    -------
    dict(method, current, percentile, samples, years, median, low, high,
         series_start, series_end, note)
    """
    out = {"method": method, "current": float("nan"), "percentile": float("nan"),
           "samples": 0, "years": 0.0, "median": float("nan"), "low": float("nan"),
           "high": float("nan"), "series_start": None, "series_end": None,
           "note": "price_per_eps_approx：以当前盈利水平回溯价格得到近似估值序列"}
    try:
        s = price_series
        if isinstance(s, pd.DataFrame):
            s = _ensure_frame(s)
            if s is None or "close" not in s.columns:
                return out
            s = pd.Series(pd.to_numeric(s["close"], errors="coerce").values,
                          index=pd.to_datetime(s.get("date", pd.RangeIndex(len(s)))))
        s = pd.Series(s).astype(float).dropna()
        if s.empty:
            return out
        try:
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            cutoff = s.index.max() - pd.Timedelta(days=365 * 5)
            s = s[s.index >= cutoff]
        except Exception:  # noqa: BLE001
            s = s.tail(1250)
        if len(s) < 30:
            out["note"] = "价格样本不足 30 个，无法计算分位"
            return out
        denom = _eps_ttm(frame) if method == "pe" else _bps_latest(frame)
        if denom is None or math.isnan(denom) or abs(denom) < 1e-9:
            out["note"] = "每股收益/每股净资产缺失或为 0，无法计算分位"
            return out
        series = s / denom
        current = float(series.iloc[-1])
        if current <= 0:      # 亏损（EPS<0）时 PE 无意义
            out["note"] = "当前每股收益为负，PE 分位不适用"
            out["current"] = current
            return out
        out.update({
            "current": current,
            "percentile": float(percentile_rank(series.values, current)),
            "samples": int(len(series)),
            "years": round(len(series) / 244.0, 2),
            "median": float(np.nanmedian(series.values)),
            "low": float(np.nanpercentile(series.values, 10)),
            "high": float(np.nanpercentile(series.values, 90)),
            "series_start": str(series.index[0].date()),
            "series_end": str(series.index[-1].date()),
        })
    except Exception as exc:  # noqa: BLE001
        out["note"] = "估值分位计算失败：%r" % exc
    return out


def fetch_valuation(symbol, force=False) -> dict:
    """估值快照 + 历史分位。

    Returns
    -------
    dict
        ``pe_ttm, pe_static, pb, ps, pcf, market_cap, float_market_cap, total_shares,
        float_shares, dividend_yield, ev_ebitda, pe_percentile, pb_percentile,
        pe_history_years, percentile_method, source``（另外附带 ``price, turnover,
        volume_ratio, eps_ttm, bps, pe_series_note`` 等便于报告展示）。

    说明
    ----
    * PE/PB 现值来自腾讯实时（GBK，字段索引实测固化于 :data:`TENCENT_FIELD`）。
    * ``pe_percentile/pb_percentile`` 为 **近似值**：历史 PE 序列无法直接获取，采用
      "盈利不变假设"（收盘价 ÷ 当前 EPS-TTM）。``percentile_method="price_per_eps_approx"``。
    * ``ev_ebitda`` 无稳定免费数据源，固定留 NaN；``ps/pcf`` 用市值 ÷ TTM 营收 / TTM 经营现金流。
    """
    code = to_ak_symbol(symbol)
    key = cache_key("valuation", code)
    try:
        if not force:
            hit = cached(key, config.RUNTIME["cache_ttl_seconds"], lambda: None)
            if isinstance(hit, dict) and hit:
                return hit
    except Exception:  # noqa: BLE001
        pass

    out = {k: float("nan") for k in
           ("pe_ttm", "pe_static", "pb", "ps", "pcf", "market_cap", "float_market_cap",
            "total_shares", "float_shares", "dividend_yield", "ev_ebitda",
            "pe_percentile", "pb_percentile")}
    out.update({"pe_history_years": 0.0, "percentile_method": "price_per_eps_approx",
                "source": "none"})
    try:
        quote = fetch_quote(symbol, force=force)
        if quote:
            out["source"] = "tencent"
            for k in ("pe_ttm", "pe_static", "pb", "market_cap", "float_market_cap",
                      "total_shares", "float_shares", "dividend_yield", "price",
                      "turnover", "volume_ratio", "amplitude", "prev_close", "name"):
                if k in quote:
                    out[k] = quote[k]
        frame = fetch_financials(symbol, force=False)
        price_series = fetch_price_series(symbol, force=False)
        eps = _eps_ttm(frame)
        bps = _bps_latest(frame)
        if (eps is None or math.isnan(eps)) and not math.isnan(_num(out.get("price"))) \
                and not math.isnan(_num(out.get("pe_ttm"))) and _num(out["pe_ttm"]) > 0:
            eps = _num(out["price"]) / _num(out["pe_ttm"])     # 用现价/PE 反推 EPS-TTM
            _note("EPS-TTM 由现价/PE 反推（财务表缺失）")
        out["eps_ttm"] = eps
        out["bps"] = bps
        # PS / PCF：市值 ÷ TTM 营收 / TTM 经营现金流
        mcap = _num(out.get("market_cap"))
        rev_ttm = _ttm_sum(frame, "营业收入")
        ocf_ttm = _ttm_sum(frame, "经营现金流")
        if not math.isnan(mcap) and not math.isnan(rev_ttm) and rev_ttm > 0:
            out["ps"] = mcap / rev_ttm
        if not math.isnan(mcap) and not math.isnan(ocf_ttm) and ocf_ttm > 0:
            out["pcf"] = mcap / ocf_ttm
        # 历史分位（近似）
        pe_pct = valuation_percentile(price_series, frame, method="pe")
        pb_pct = valuation_percentile(price_series, frame, method="pb")
        out["pe_percentile"] = pe_pct.get("percentile", float("nan"))
        out["pb_percentile"] = pb_pct.get("percentile", float("nan"))
        out["pe_history_years"] = pe_pct.get("years", 0.0)
        out["pe_series_note"] = pe_pct.get("note", "")
        out["pb_series_note"] = pb_pct.get("note", "")
        out["pe_median_5y"] = pe_pct.get("median", float("nan"))
        out["pb_median_5y"] = pb_pct.get("median", float("nan"))
        if not math.isnan(_num(out["pe_percentile"])):
            _note("PE 历史分位 %.0f%%（近似法：收盘价/当前EPS-TTM，窗口 %.1f 年）"
                  % (_num(out["pe_percentile"]) * 100, _num(out["pe_history_years"])))
        # 行业中位数（尽力而为，失败不影响主流程）
        peers = fetch_industry_peers(symbol, force=False)
        if isinstance(peers, pd.DataFrame) and not peers.empty:
            attrs = peers.attrs or {}
            out["industry_pe_median"] = attrs.get("pe_median", float("nan"))
            out["industry_pb_median"] = attrs.get("pb_median", float("nan"))
            out["industry_name"] = attrs.get("board_name")
            out["industry_peers_n"] = len(peers)
    except Exception as exc:  # noqa: BLE001
        _note("fetch_valuation 异常降级：%r" % exc)
    if out.get("source") != "none":
        try:
            from .utils import cache_put
            cache_put(key, out)
        except Exception:  # noqa: BLE001
            pass
    return out


# --------------------------------------------------------------------------- #
# 3. 同行业对比
# --------------------------------------------------------------------------- #
def _industry_board(symbol: str) -> tuple:
    """返回 (板块代码, 板块名)。优先东财财务快报给出的 EM 行业分类，其次个股所属板块。"""
    code, name = "", ""
    try:
        url = (_EM_DATACENTER + "?reportName=RPT_LICO_FN_CPD&columns=SECUCODE,BOARD_CODE,"
               "BOARD_NAME,PUBLISHNAME&filter=(SECUCODE=\"%s\")&pageSize=1"
               "&sortColumns=REPORTDATE&sortTypes=-1" % _secucode(symbol))
        payload = retry_call(lambda: http_get(url, json_body=True), attempts=2, delay=0.6,
                             default=None, label="board_code")
        rows = ((payload or {}).get("result") or {}).get("data") or []
        if rows:
            code = str(rows[0].get("BOARD_CODE") or "")
            name = str(rows[0].get("BOARD_NAME") or rows[0].get("PUBLISHNAME") or "")
            if code.startswith("BK"):
                return code, name
    except Exception as exc:  # noqa: BLE001
        _note("行业板块代码查询失败：%r" % exc)
    # 备选：个股所属板块列表（slist，本机实测返回 dict 结构的 diff）
    try:
        url = (_EM_SLIST + "?secid=%s&spt=3&fields=f12,f13,f14,f3,f62&po=1&pn=1&pz=60"
               "&fid=f62" % _secid(symbol))
        payload = retry_call(lambda: http_get(url, json_body=True), attempts=3, delay=0.8,
                             default=None, label="slist_board")
        diff = ((payload or {}).get("data") or {}).get("diff")
        items = list(diff.values()) if isinstance(diff, dict) else list(diff or [])
        boards = [it for it in items if str(it.get("f12", "")).startswith("BK")]
        if boards:
            # 行业特征优先：名字含「Ⅱ/Ⅲ/行业/制造/设备/服务」等，退而取首个
            pref = [b for b in boards if any(k in str(b.get("f14", ""))
                                             for k in ("Ⅱ", "Ⅲ", "行业", "设备", "制造"))]
            pick = (pref or boards)[0]
            return str(pick.get("f12")), str(pick.get("f14"))
    except Exception as exc:  # noqa: BLE001
        _note("所属板块列表查询失败：%r" % exc)
    return "", ""


def fetch_industry_peers(symbol, force=False) -> pd.DataFrame:
    """同行业公司 PE/PB/涨跌幅横向对比表。

    Returns
    -------
    pandas.DataFrame
        列 ``代码, 名称, PE, PB, 涨跌幅, 总市值``（按涨跌幅降序）。
        ``df.attrs`` 附带 ``board_code, board_name, pe_median, pb_median, n``。
        板块接口失效时返回空 DataFrame（**不影响其它因子**）。

    备注
    ----
    板块代码取东财 F10 行业分类（实测 600519 → ``BK1277 白酒Ⅱ``）；成分股来自
    ``push2.eastmoney.com/api/qt/clist/get``（``f9``=PE动, ``f23``=PB, ``f12``=代码,
    ``f14``=名称, ``f3``=涨跌幅, ``f20``=总市值）。该域名在本机**间歇性**
    ``RemoteDisconnected``，故用 :func:`core.utils.retry_call` 重试 3 次。
    """
    code = to_ak_symbol(symbol)
    key = cache_key("peers", code)

    def _producer() -> pd.DataFrame:
        board_code, board_name = _industry_board(symbol)
        if not board_code:
            _note("未取到行业板块代码，同行对比数据不可用")
            return pd.DataFrame()
        url = (_EM_CLIST + "?pn=1&pz=60&po=1&np=1&fltt=2&invt=2&fid=f3&fs=b:%s"
               "&fields=f12,f14,f9,f23,f3,f20" % board_code)
        payload = retry_call(lambda: http_get(url, json_body=True), attempts=3, delay=0.9,
                             default=None, label="clist_peers")
        diff = ((payload or {}).get("data") or {}).get("diff") or []
        items = list(diff.values()) if isinstance(diff, dict) else list(diff)
        if not items:
            _note("行业板块 %s(%s) 成分股取数失败（push2 域名间歇性拒连）"
                  % (board_code, board_name))
            return pd.DataFrame()
        rows = [{"代码": it.get("f12"), "名称": it.get("f14"),
                 "PE": _num(it.get("f9")), "PB": _num(it.get("f23")),
                 "涨跌幅": _num(it.get("f3")), "总市值": _num(it.get("f20"))}
                for it in items]
        df = pd.DataFrame(rows).sort_values("涨跌幅", ascending=False).reset_index(drop=True)
        pe_valid = df["PE"].where(df["PE"] > 0)
        df.attrs.update({
            "board_code": board_code, "board_name": board_name, "n": int(len(df)),
            "pe_median": float(pe_valid.median()) if pe_valid.notna().any() else float("nan"),
            "pb_median": float(df["PB"].where(df["PB"] > 0).median())
            if df["PB"].notna().any() else float("nan"),
        })
        _note("同行对比：%s(%s) 共 %d 家，PE 中位数 %.1f"
              % (board_name, board_code, len(df), _num(df.attrs["pe_median"])))
        return df

    try:
        if force:
            return _producer()
        out = cached(key, config.RUNTIME["cache_ttl_daily"], _producer)
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        _note("fetch_industry_peers 异常降级：%r" % exc)
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# 4. 财务派生分析
# --------------------------------------------------------------------------- #
def _row(frame, offset: int = -1) -> dict:
    """取第 offset 行（默认最新一期）为 dict；越界返回 {}。"""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    try:
        df = frame.sort_values("报告期")
        return dict(df.iloc[offset])
    except Exception:  # noqa: BLE001
        return {}


def _vals(frame, col: str, n: int = 6) -> list:
    """取某列最近 n 期（升序）的有效值列表。"""
    if frame is None or not isinstance(frame, pd.DataFrame) or col not in frame.columns:
        return []
    try:
        s = pd.to_numeric(frame.sort_values("报告期")[col], errors="coerce").dropna()
        return [float(x) for x in s.tail(n).tolist()]
    except Exception:  # noqa: BLE001
        return []


def _slope(vals: list) -> float:
    """最小二乘斜率（每期变化量），样本 <2 返回 NaN。"""
    if not vals or len(vals) < 2:
        return float("nan")
    x = np.arange(len(vals), dtype=float)
    y = np.asarray(vals, dtype=float)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:  # noqa: BLE001
        return float("nan")


def growth_quality_score(frame) -> dict:
    """成长质量评分（0~100）与分项拆解。

    Returns
    -------
    dict
        ``{"score","growth","quality","cashflow","safety","detail":[...]}``，
        四个分项均为 0~100；``detail`` 是中文说明列表。
    """
    out = {"score": 50.0, "growth": 50.0, "quality": 50.0, "cashflow": 50.0,
           "safety": 50.0, "detail": []}
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        out["detail"].append("无财务数据，评分取中性 50 分")
        return out
    last = _row(frame, -1)
    prev = _row(frame, -2)

    # ---- 成长：营收/净利同比 + 环比加速度 -------------------------------- #
    rev_yoy = _num(last.get("营收同比"))
    np_yoy = _num(last.get("净利同比"))
    parts, g = [], 50.0
    if not math.isnan(rev_yoy):
        g += clamp(rev_yoy / 40.0, -1, 1) * 18
        parts.append("营收同比 %.1f%%" % rev_yoy)
    if not math.isnan(np_yoy):
        g += clamp(np_yoy / 40.0, -1, 1) * 22
        parts.append("净利同比 %.1f%%" % np_yoy)
    accel = _num(last.get("净利环比")) - _num(prev.get("净利环比"))
    if not math.isnan(accel):
        g += clamp(accel / 50.0, -1, 1) * 10
        parts.append("净利环比%s %.1f 个百分点"
                     % ("加速" if accel > 0 else "减速", abs(accel)))
    out["growth"] = clamp(g, 0, 100)
    out["detail"].append("成长性 %d 分：" % round(out["growth"]) + "，".join(parts or ["数据不足"]))

    # ---- 质量：ROE 水平/趋势 + 毛利率趋势 + 扣非占比 --------------------- #
    roe = _num(last.get("ROE"))
    roe_vals = _vals(frame, "ROE", 6)
    q = 50.0
    parts = []
    if not math.isnan(roe):
        q += clamp((roe - 8.0) / 15.0, -1, 1) * 22
        parts.append("ROE %.1f%%" % roe)
    if len(roe_vals) >= 3:
        sl = _slope(roe_vals[-4:])
        if not math.isnan(sl):
            q += clamp(sl / 1.5, -1, 1) * 12
            parts.append("ROE %s" % ("上行" if sl > 0 else "下行"))
    gm_vals = _vals(frame, "毛利率", 6)
    if len(gm_vals) >= 3:
        sl = _slope(gm_vals[-4:])
        if not math.isnan(sl):
            q += clamp(sl / 2.0, -1, 1) * 8
            parts.append("毛利率%s" % ("走高" if sl > 0 else "走低"))
    ded = _num(last.get("扣非占比"))
    if not math.isnan(ded):
        q += clamp((ded - 85.0) / 25.0, -1, 1) * 8
        parts.append("扣非占比 %.1f%%" % ded)
    out["quality"] = clamp(q, 0, 100)
    out["detail"].append("质量 %d 分：" % round(out["quality"]) + "，".join(parts or ["数据不足"]))

    # ---- 现金流：经营现金流/净利润 --------------------------------------- #
    c = 50.0
    parts = []
    ratio = _num(last.get("经营现金流净利比"))
    if math.isnan(ratio):
        ocf, ni = _num(last.get("经营现金流")), _num(last.get("净利润"))
        if not math.isnan(ocf) and not math.isnan(ni) and abs(ni) > 1e-9:
            ratio = ocf / ni
    if not math.isnan(ratio):
        c += clamp((ratio - 1.0) / 0.8, -1, 1) * 28
        parts.append("经营现金流/净利润 %.2f" % ratio)
    ocf_yoy = _num(last.get("经营现金流同比"))
    if not math.isnan(ocf_yoy):
        c += clamp(ocf_yoy / 40.0, -1, 1) * 14
        parts.append("现金流同比 %.1f%%" % ocf_yoy)
    ps_ocf = _num(last.get("每股经营现金流"))
    if not math.isnan(ps_ocf):
        c += clamp(ps_ocf / 3.0, -1, 1) * 8
        parts.append("每股经营现金流 %.2f 元" % ps_ocf)
    out["cashflow"] = clamp(c, 0, 100)
    out["detail"].append("现金流 %d 分：" % round(out["cashflow"]) + "，".join(parts or ["数据不足"]))

    # ---- 安全：资产负债率 + 流动比率 ------------------------------------- #
    s_ = 50.0
    parts = []
    debt = _num(last.get("资产负债率"))
    if not math.isnan(debt):
        s_ += clamp((50.0 - debt) / 35.0, -1, 1) * 26
        parts.append("资产负债率 %.1f%%" % debt)
    cr = _num(last.get("流动比率"))
    if not math.isnan(cr):
        s_ += clamp((cr - 1.2) / 1.0, -1, 1) * 16
        parts.append("流动比率 %.2f" % cr)
    qr = _num(last.get("速动比率"))
    if not math.isnan(qr):
        s_ += clamp((qr - 0.9) / 0.8, -1, 1) * 8
        parts.append("速动比率 %.2f" % qr)
    out["safety"] = clamp(s_, 0, 100)
    out["detail"].append("安全性 %d 分：" % round(out["safety"]) + "，".join(parts or ["数据不足"]))

    out["score"] = round(clamp(0.35 * out["growth"] + 0.28 * out["quality"]
                               + 0.22 * out["cashflow"] + 0.15 * out["safety"], 0, 100), 1)
    out["detail"].append("综合成长质量得分 %.1f/100（权重：成长 35%%、质量 28%%、"
                         "现金流 22%%、安全 15%%）" % out["score"])
    return out


def earnings_catalyst(frame, news=None) -> list:
    """识别财报/事件催化剂。

    Returns
    -------
    list[dict]
        ``[{"date","type","title","direction","strength","detail"}]``，
        type ∈ {``业绩超预期``, ``环比加速``, ``扭亏为盈``, ``业绩下滑``,
        ``亏损``, ``订单/新产品``, ``重大资产重组``}。
    """
    events: list = []
    if frame is not None and isinstance(frame, pd.DataFrame) and not frame.empty:
        last = _row(frame, -1)
        prev = _row(frame, -2)
        date = str(last.get("报告期") or "")
        rev_yoy, np_yoy = _num(last.get("营收同比")), _num(last.get("净利同比"))
        np_qoq, prev_qoq = _num(last.get("净利环比")), _num(prev.get("净利环比"))
        if not math.isnan(np_yoy) and np_yoy > 30 and not math.isnan(np_qoq) \
                and (math.isnan(prev_qoq) or np_qoq >= prev_qoq):
            events.append({"date": date, "type": "业绩超预期",
                           "title": "%s 净利同比 +%.1f%% 且环比加速" % (date, np_yoy),
                           "direction": 1, "strength": clamp(0.45 + np_yoy / 150.0, 0.4, 0.9),
                           "detail": "净利润同比 %.1f%%，环比 %.1f%%，增长动能增强"
                                     % (np_yoy, np_qoq)})
        elif not math.isnan(np_yoy) and np_yoy > 20:
            events.append({"date": date, "type": "业绩超预期",
                           "title": "%s 净利同比 +%.1f%%" % (date, np_yoy),
                           "direction": 1, "strength": clamp(0.3 + np_yoy / 200.0, 0.3, 0.7),
                           "detail": "净利润同比高增长 %.1f%%" % np_yoy})
        if not math.isnan(np_yoy) and np_yoy < -20:
            events.append({"date": date, "type": "业绩下滑",
                           "title": "%s 净利同比 %.1f%%" % (date, np_yoy),
                           "direction": -1, "strength": clamp(0.3 + abs(np_yoy) / 150.0, 0.3, 0.85),
                           "detail": "净利润同比下滑 %.1f%%，基本面走弱" % abs(np_yoy)})
        # 扭亏为盈：单季净利由负转正
        cur_q, prev_q = _num(last.get("单季净利")), _num(prev.get("单季净利"))
        if not math.isnan(cur_q) and not math.isnan(prev_q) and prev_q < 0 <= cur_q:
            events.append({"date": date, "type": "扭亏为盈",
                           "title": "%s 单季净利转正" % date, "direction": 1,
                           "strength": 0.7,
                           "detail": "单季净利润由 %.2f 转为 %.2f，业绩拐点" % (prev_q, cur_q)})
        if not math.isnan(_num(last.get("净利润"))) and _num(last.get("净利润")) < 0:
            events.append({"date": date, "type": "亏损",
                           "title": "%s 净利润为负" % date, "direction": -1, "strength": 0.6,
                           "detail": "最新报告期净利润为负，估值指标失真"})
    # ---- 新闻标题关键词 -------------------------------------------------- #
    order_kw = ("订单", "中标", "签订", "合同", "采购", "量产", "投产", "获批", "提价",
                "涨价", "新产品", "新品", "扩产", "满产")
    restr_kw = ("重组", "并购", "收购", "资产注入", "借壳", "合并", "混改")
    for it in (news or []):
        if not isinstance(it, dict):
            continue
        title = str(it.get("title") or "")
        date = str(it.get("date") or "")
        if any(k in title for k in restr_kw):
            events.append({"date": date, "type": "重大资产重组", "title": title,
                           "direction": 1, "strength": 0.7,
                           "detail": "涉及重组/并购类事项，事件驱动型利好（需关注整合风险）"})
        elif any(k in title for k in order_kw):
            events.append({"date": date, "type": "订单/新产品", "title": title,
                           "direction": 1, "strength": 0.5,
                           "detail": "订单/新品/产能类消息，利好中期业绩"})
    return events


# --------------------------------------------------------------------------- #
# 5. 因子生成
# --------------------------------------------------------------------------- #
def _growth_factors(frame) -> list:
    fs: list = []
    last, prev = _row(frame, -1), _row(frame, -2)
    rev_yoy, np_yoy = _num(last.get("营收同比")), _num(last.get("净利同比"))
    rev_qoq, np_qoq = _num(last.get("营收环比")), _num(last.get("净利环比"))
    prev_np_qoq = _num(prev.get("净利环比"))
    horizon = ("d20", "m1", "q1", "h1", "y1")
    if math.isnan(rev_yoy) and math.isnan(np_yoy):
        fs.append(_neutral("fundamental_growth", "业绩成长性",
                           "缺少财务数据，成长性无法评估", 0.15,
                           theory="PEG/成长股理论", horizon=horizon))
    else:
        score = 0.0
        if not math.isnan(rev_yoy):
            score += math.tanh(rev_yoy / 30.0) * 0.45
        if not math.isnan(np_yoy):
            score += math.tanh(np_yoy / 30.0) * 0.55
        # 环比加速 / 减速
        accel = None
        if not math.isnan(np_qoq):
            if not math.isnan(prev_np_qoq):
                accel = np_qoq - prev_np_qoq
                score += clamp(accel / 40.0, -1, 1) * 0.35
            else:
                score += clamp(np_qoq / 60.0, -1, 1) * 0.2
        # 连续多季改善（最近 4 期营收同比是否递增）
        yoy_hist = _vals(frame, "营收同比", 4)
        improving = len(yoy_hist) >= 3 and all(
            yoy_hist[i] <= yoy_hist[i + 1] + 1e-9 for i in range(len(yoy_hist) - 1))
        if improving:
            score += 0.15
        direction = DIR_BULL if score > 0.05 else (DIR_BEAR if score < -0.05 else DIR_NEUTRAL)
        detail = "营收同比 %s、净利同比 %s" % (
            "%.1f%%" % rev_yoy if not math.isnan(rev_yoy) else "缺失",
            "%.1f%%" % np_yoy if not math.isnan(np_yoy) else "缺失")
        if accel is not None:
            detail += "，净利环比%s %.1f 个百分点" % ("加速" if accel > 0 else "减速", abs(accel))
        if improving:
            detail += "，连续多季营收增速改善"
        fs.append(Factor("fundamental_growth", "业绩成长性", Category.FUND, direction,
                         clamp(abs(score), 0, 1), _reliability(frame, 0.85, len(frame)),
                         value={"营收同比": rev_yoy, "净利同比": np_yoy, "环比增速": np_qoq},
                         detail=detail, theory="成长股理论 / PEG", horizon=horizon))
    # ---- 营收与净利增速背离 --------------------------------------------- #
    if not math.isnan(rev_yoy) and not math.isnan(np_yoy):
        gap = np_yoy - rev_yoy
        if gap < -10:      # 增收不增利
            fs.append(Factor("growth_divergence", "增收不增利", Category.FUND, DIR_BEAR,
                             clamp(abs(gap) / 60.0, 0.2, 0.85),
                             _reliability(frame, 0.7, len(frame)),
                             value=gap, detail="净利增速低于营收增速 %.1f 个百分点，"
                                               "毛利率或费用端承压" % abs(gap),
                             theory="杜邦分析", horizon=horizon))
        elif gap > 15:     # 增利不增收（含一次性收益可能）
            fs.append(Factor("growth_divergence", "利润增速远超营收", Category.FUND,
                             DIR_NEUTRAL, clamp(abs(gap) / 80.0, 0.15, 0.5),
                             _reliability(frame, 0.5, len(frame)), value=gap,
                             detail="净利增速高于营收 %.1f 个百分点，需警惕非经常性损益"
                                    "（扣非占比 %s）" % (abs(gap),
                                                   _num(last.get("扣非占比"))),
                             theory="利润含金量", horizon=horizon))
    return fs


def _quality_factors(frame) -> list:
    fs: list = []
    last = _row(frame, -1)
    horizon = ("d20", "m1", "q1", "h1", "y1")
    # ---- ROE ------------------------------------------------------------- #
    roe = _num(last.get("ROE"))
    roe_vals = _vals(frame, "ROE", 6)
    if math.isnan(roe):
        fs.append(_neutral("roe_quality", "ROE 水平与趋势", "缺少 ROE 数据", 0.15,
                           theory="杜邦分析", horizon=horizon))
    else:
        sl = _slope(roe_vals[-4:]) if len(roe_vals) >= 3 else float("nan")
        score = math.tanh((roe - 8.0) / 12.0) * 0.75
        if not math.isnan(sl):
            score += clamp(sl / 1.5, -1, 1) * 0.25
        detail = "ROE %.2f%%（%s）" % (roe, "高位" if roe > 15 else
                                      ("中等" if roe > 8 else "偏低"))
        if not math.isnan(sl):
            detail += "，趋势%s" % ("上行" if sl > 0.05 else ("下行" if sl < -0.05 else "走平"))
        fs.append(Factor("roe_quality", "ROE 水平与趋势", Category.FUND,
                         DIR_BULL if score > 0.05 else (DIR_BEAR if score < -0.05 else 0),
                         clamp(abs(score), 0, 1), _reliability(frame, 0.85, len(frame)),
                         value=roe, detail=detail, theory="杜邦分析 / 巴菲特 ROE 选股",
                         horizon=horizon))
    # ---- 毛利率/净利率趋势 ----------------------------------------------- #
    gm, nm = _vals(frame, "毛利率", 6), _vals(frame, "净利率", 6)
    gm_sl = _slope(gm[-4:]) if len(gm) >= 3 else float("nan")
    nm_sl = _slope(nm[-4:]) if len(nm) >= 3 else float("nan")
    if math.isnan(gm_sl) and math.isnan(nm_sl):
        fs.append(_neutral("margin_trend", "毛利率/净利率趋势", "缺少利润率序列", 0.15,
                           theory="盈利能力分析", horizon=horizon))
    else:
        score = 0.0
        if not math.isnan(gm_sl):
            score += clamp(gm_sl / 1.5, -1, 1) * 0.55
        if not math.isnan(nm_sl):
            score += clamp(nm_sl / 1.2, -1, 1) * 0.45
        detail = "毛利率 %s，净利率 %s" % (
            "%.1f%%(%+.2fpp/期)" % (gm[-1], gm_sl) if gm and not math.isnan(gm_sl) else "缺失",
            "%.1f%%(%+.2fpp/期)" % (nm[-1], nm_sl) if nm and not math.isnan(nm_sl) else "缺失")
        fs.append(Factor("margin_trend", "毛利率/净利率趋势", Category.FUND,
                         DIR_BULL if score > 0.05 else (DIR_BEAR if score < -0.05 else 0),
                         clamp(abs(score), 0, 1), _reliability(frame, 0.8, len(frame)),
                         value={"毛利率": gm[-1] if gm else None,
                                "净利率": nm[-1] if nm else None},
                         detail=detail, theory="盈利能力分析", horizon=horizon))
    # ---- 扣非占比（利润含金量） ----------------------------------------- #
    ded = _num(last.get("扣非占比"))
    if math.isnan(ded):
        fs.append(_neutral("profit_quality", "利润含金量", "缺少扣非净利润数据", 0.15,
                           theory="利润含金量", horizon=horizon))
    else:
        score = clamp((ded - 80.0) / 20.0, -1, 1)
        fs.append(Factor("profit_quality", "利润含金量", Category.FUND,
                         DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.2 else 0),
                         clamp(abs(score), 0, 1), _reliability(frame, 0.8, len(frame)),
                         value=ded, detail="扣非净利润占净利润 %.1f%%，%s"
                                           % (ded, "利润质量高" if ded > 90 else
                                              ("含一定非经常性损益" if ded > 70 else
                                               "非经常性损益占比偏高")),
                         theory="利润含金量", horizon=horizon))
    return fs


def _cashflow_factors(frame) -> list:
    horizon = ("d20", "m1", "q1", "h1", "y1")
    last = _row(frame, -1)
    ratio = _num(last.get("经营现金流净利比"))
    if math.isnan(ratio):
        ocf, ni = _num(last.get("经营现金流")), _num(last.get("净利润"))
        if not math.isnan(ocf) and not math.isnan(ni) and abs(ni) > 1e-9:
            ratio = ocf / ni
    ocf_yoy = _num(last.get("经营现金流同比"))
    ps_ocf = _num(last.get("每股经营现金流"))
    if math.isnan(ratio) and math.isnan(ocf_yoy) and math.isnan(ps_ocf):
        return [_neutral("cashflow_health", "现金流健康度", "缺少现金流数据", 0.15,
                         theory="现金流分析", horizon=horizon)]
    score, parts = 0.0, []
    if not math.isnan(ratio):
        score += clamp((ratio - 1.0) / 0.8, -1, 1) * 0.5
        parts.append("经营现金流/净利润 %.2f" % ratio)
    if not math.isnan(ocf_yoy):
        score += clamp(ocf_yoy / 40.0, -1, 1) * 0.3
        parts.append("现金流同比 %.1f%%" % ocf_yoy)
    if not math.isnan(ps_ocf):
        score += clamp(ps_ocf / 3.0, -1, 1) * 0.2
        parts.append("每股经营现金流 %.2f 元" % ps_ocf)
    detail = "，".join(parts)
    if not math.isnan(ratio):
        detail += "；" + ("现金流覆盖利润良好" if ratio > 1 else "现金流弱于利润，需关注回款")
    return [Factor("cashflow_health", "现金流健康度", Category.FUND,
                   DIR_BULL if score > 0.05 else (DIR_BEAR if score < -0.05 else 0),
                   clamp(abs(score), 0, 1), _reliability(frame, 0.85, len(frame)),
                   value={"现金流净利比": ratio, "现金流同比": ocf_yoy},
                   detail=detail, theory="现金流分析 / 收付实现制", horizon=horizon)]


def _leverage_factors(frame) -> list:
    horizon = ("d20", "m1", "q1", "h1", "y1")
    last = _row(frame, -1)
    debt = _num(last.get("资产负债率"))
    cr, qr = _num(last.get("流动比率")), _num(last.get("速动比率"))
    debt_vals = _vals(frame, "资产负债率", 6)
    if math.isnan(debt) and math.isnan(cr):
        return [_neutral("leverage_risk", "负债与安全边际", "缺少负债数据", 0.15,
                         theory="财务风险分析", horizon=horizon)]
    score, parts = 0.0, []
    if not math.isnan(debt):
        score += clamp((50.0 - debt) / 30.0, -1, 1) * 0.5
        parts.append("资产负债率 %.1f%%" % debt)
        if debt > 70:
            parts.append("负债偏高")
    if not math.isnan(cr):
        score += clamp((cr - 1.2) / 1.0, -1, 1) * 0.3
        parts.append("流动比率 %.2f" % cr)
    if not math.isnan(qr):
        score += clamp((qr - 0.9) / 0.8, -1, 1) * 0.2
        parts.append("速动比率 %.2f" % qr)
    # 有息负债压力：以资产负债率上升 + 流动比率下降的组合近似（无明细科目时的代理）
    if len(debt_vals) >= 3:
        sl = _slope(debt_vals[-4:])
        if not math.isnan(sl) and sl > 0.3:
            score -= 0.15
            parts.append("负债率上行 %.2fpp/期" % sl)
    return [Factor("leverage_risk", "负债与安全边际", Category.FUND,
                   DIR_BULL if score > 0.05 else (DIR_BEAR if score < -0.05 else 0),
                   clamp(abs(score), 0, 1), _reliability(frame, 0.8, len(frame)),
                   value={"资产负债率": debt, "流动比率": cr},
                   detail="，".join(parts) + "（有息负债明细缺失，以负债率+流动比率代理）",
                   theory="财务风险分析", horizon=horizon)]


def _valuation_factors(frame, valuation, price) -> list:
    fs: list = []
    val = valuation if isinstance(valuation, dict) else {}
    horizon = ("d20", "m1", "q1", "h1", "y1")
    pe = _num(val.get("pe_ttm"))
    pb = _num(val.get("pb"))
    pe_pct = _num(val.get("pe_percentile"))
    pb_pct = _num(val.get("pb_percentile"))
    approx = val.get("percentile_method") == "price_per_eps_approx"

    # ---- PE/PB 历史分位（低分位看涨） ------------------------------------ #
    if math.isnan(pe_pct) and math.isnan(pb_pct):
        fs.append(_neutral("valuation_percentile", "估值历史分位",
                           "无法计算 PE/PB 历史分位（缺少价格序列或每股收益）", 0.2,
                           theory="估值分位回归", horizon=horizon))
    else:
        parts, score = [], 0.0
        if not math.isnan(pe_pct):
            score += (0.5 - pe_pct) * 2.0 * 0.6
            parts.append("PE %.1f（近 %.1f 年分位 %.0f%%）"
                         % (pe, _num(val.get("pe_history_years")), pe_pct * 100))
        if not math.isnan(pb_pct):
            score += (0.5 - pb_pct) * 2.0 * 0.4
            parts.append("PB %.2f（分位 %.0f%%）" % (pb, pb_pct * 100))
        rel = 0.75 if approx else 0.85
        fs.append(Factor("valuation_percentile", "估值历史分位", Category.FUND,
                         DIR_BULL if score > 0.05 else (DIR_BEAR if score < -0.05 else 0),
                         clamp(abs(score), 0, 1), _reliability(frame, rel, 5),
                         value={"pe_ttm": pe, "pe_percentile": pe_pct,
                                "pb_percentile": pb_pct},
                         detail="，".join(parts) + ("；分位为近似值（盈利不变假设，"
                                                    "收盘价÷当前 EPS-TTM）" if approx else ""),
                         theory="估值分位回归 / 均值回复", horizon=horizon))

    # ---- PEG ------------------------------------------------------------- #
    np_yoy = _num(_row(frame, -1).get("净利同比"))
    if math.isnan(pe) or pe <= 0 or math.isnan(np_yoy):
        fs.append(_neutral("peg_level", "PEG 估值", "缺少 PE 或净利增速，PEG 不适用", 0.2,
                           theory="PEG 估值", horizon=horizon))
    elif np_yoy <= 0:
        fs.append(Factor("peg_level", "PEG 估值", Category.FUND, DIR_BEAR, 0.4,
                         _reliability(frame, 0.35, len(frame)),
                         value=None, detail="净利润同比为负（%.1f%%），PEG 无意义，"
                                            "估值缺乏成长支撑" % np_yoy,
                         theory="PEG 估值", horizon=horizon))
    else:
        peg = pe / np_yoy
        score = clamp((1.2 - peg) / 1.2, -1, 1)
        # 周期股 PE 低/高都不能简单套 PEG → 降低可信度
        rel = 0.8
        ind = str(val.get("industry_name") or "")
        if any(k in ind for k in CYCLICAL_KEYWORDS):
            rel = 0.45
        fs.append(Factor("peg_level", "PEG 估值", Category.FUND,
                         DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.2 else 0),
                         clamp(abs(score), 0, 1), _reliability(frame, rel, len(frame)),
                         value=peg, detail="PEG = PE(TTM) %.1f ÷ 净利增速 %.1f%% = %.2f，%s"
                                           % (pe, np_yoy, peg,
                                              "估值相对成长性便宜" if peg < 1 else
                                              ("估值与成长匹配" if peg < 1.5 else
                                               "估值偏贵（成长性不足以支撑）")),
                         theory="PEG 估值（彼得·林奇）", horizon=horizon))

    # ---- 相对行业估值 ---------------------------------------------------- #
    ind_pe = _num(val.get("industry_pe_median"))
    if math.isnan(ind_pe) or math.isnan(pe) or pe <= 0:
        fs.append(_neutral("relative_valuation", "相对行业估值",
                           "缺少行业中位数（行业板块接口不可用），无法横向对比", 0.2,
                           theory="相对估值", horizon=horizon))
    else:
        ratio = pe / ind_pe if ind_pe > 0 else float("nan")
        score = clamp((1.0 - ratio) / 0.6, -1, 1) if not math.isnan(ratio) else 0.0
        fs.append(Factor("relative_valuation", "相对行业估值", Category.FUND,
                         DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.1 else 0),
                         clamp(abs(score), 0, 1), _reliability(frame, 0.6, 5),
                         value=ratio,
                         detail="PE %.1f vs 行业中位数 %.1f（%s %.0f%%），%s"
                                % (pe, ind_pe, str(val.get("industry_name") or "行业"),
                                   abs(ratio - 1) * 100,
                                   "低于行业中位，估值有相对优势" if ratio < 1
                                   else "高于行业中位，需成长性支撑"),
                         theory="相对估值", horizon=horizon))
    return fs


def _sustainability_factors(frame, name: str = "") -> list:
    """成长可持续性：应收/存货周转异常、研发费用率、亏损警示。"""
    fs: list = []
    horizon = ("d20", "m1", "q1", "h1", "y1")
    last = _row(frame, -1)
    recv = _vals(frame, "应收账款周转率", 6)
    inv = _vals(frame, "存货周转率", 6)
    rev_yoy = _num(last.get("营收同比"))
    parts, score = [], 0.0
    if len(recv) >= 3:
        sl = _slope(recv[-4:])
        if not math.isnan(sl) and recv[-1] > 0:
            pct = sl / abs(recv[-1]) * 100
            if pct < -5:
                score -= clamp(abs(pct) / 30.0, 0.1, 0.8)
                parts.append("应收账款周转率下降 %.1f%%/期（回款变慢/放宽信用）" % abs(pct))
            elif pct > 5:
                score += clamp(pct / 30.0, 0.05, 0.4)
                parts.append("应收账款周转率提升 %.1f%%/期" % pct)
    if len(inv) >= 3:
        sl = _slope(inv[-4:])
        if not math.isnan(sl) and inv[-1] > 0:
            pct = sl / abs(inv[-1]) * 100
            if pct < -5:
                score -= clamp(abs(pct) / 40.0, 0.05, 0.5)
                parts.append("存货周转率下降 %.1f%%/期（可能积压）" % abs(pct))
    rnd = _num(last.get("研发费用率"))
    if not math.isnan(rnd):
        parts.append("研发费用率 %.2f%%" % rnd)
        if rnd > 5:
            score += 0.15
        elif rnd < 1:
            score -= 0.05
    if not math.isnan(rev_yoy) and rev_yoy > 0 and score < -0.3:
        parts.append("营收增速 %.1f%% 与营运效率恶化同时出现，需警惕收入质量" % rev_yoy)
    if not parts:
        fs.append(_neutral("receivable_risk", "成长可持续性（应收/存货/研发）",
                           "缺少应收/存货/研发数据", 0.2, theory="营运能力分析",
                           horizon=horizon))
    else:
        fs.append(Factor("receivable_risk", "成长可持续性（应收/存货/研发）", Category.FUND,
                         DIR_BULL if score > 0.08 else (DIR_BEAR if score < -0.08 else 0),
                         clamp(abs(score), 0, 1), _reliability(frame, 0.7, len(frame)),
                         value={"应收周转": recv[-1] if recv else None,
                                "存货周转": inv[-1] if inv else None,
                                "研发费用率": rnd},
                         detail="；".join(parts), theory="营运能力分析", horizon=horizon))
    # ---- 亏损警示 -------------------------------------------------------- #
    ni = _num(last.get("净利润"))
    if not math.isnan(ni) and ni < 0:
        fs.append(Factor("loss_warning", "亏损警示", Category.FUND, DIR_BEAR, 0.75,
                         _reliability(frame, 0.9, len(frame)), value=ni,
                         detail="最新报告期净利润为负（%.2f），PE/PEG 等估值指标失真，"
                                "估值分位与 PEG 可信度已下调" % ni,
                         theory="财务困境理论", horizon=("d20", "m1", "q1", "h1")))
    return fs


def _event_factors(frame, news) -> list:
    """催化剂类因子：业绩超预期 / 扭亏 / 重组。"""
    fs: list = []
    events = earnings_catalyst(frame, news)
    if not events:
        fs.append(_neutral("earnings_catalyst", "业绩催化剂", "未识别到明显业绩催化剂", 0.3,
                           theory="事件驱动", horizon=("q1", "h1", "y1")))
        return fs
    # 取方向最强的事件作为 earnings_catalyst
    main = max(events, key=lambda e: e.get("strength", 0))
    direction = int(main.get("direction", 0))
    fs.append(Factor("earnings_catalyst", "业绩催化剂", Category.FUND, direction,
                     clamp(main.get("strength", 0.3), 0, 1),
                     _reliability(frame, 0.75, len(frame)),
                     value=main.get("type"), detail="%s：%s"
                     % (main.get("type"), main.get("detail") or main.get("title")),
                     theory="事件驱动 / 盈余惊喜", horizon=("q1", "h1", "y1")))
    for e in events:
        if e.get("type") == "扭亏为盈":
            fs.append(Factor("turnaround", "扭亏为盈", Category.FUND, DIR_BULL,
                             clamp(e.get("strength", 0.6), 0, 1),
                             _reliability(frame, 0.7, len(frame)), value=e.get("date"),
                             detail=e.get("detail"), theory="困境反转",
                             horizon=("q1", "h1", "y1")))
        if e.get("type") == "重大资产重组":
            fs.append(Factor("restructure_event", "重大资产重组", Category.FUND, DIR_BULL,
                             clamp(e.get("strength", 0.6), 0, 1), 0.5,
                             value=e.get("title"), detail=e.get("detail"),
                             theory="事件驱动 / 重组预期",
                             horizon=("d20", "m1", "q1", "h1")))
    return fs


def fundamental_factors(symbol, frame=None, valuation=None, price=None, name="",
                        news=None) -> list:
    """产出基本面因子列表。

    Parameters
    ----------
    symbol : str
        股票代码（仅用于日志/展示）。
    frame : pd.DataFrame | None
        :func:`fetch_financials` 的产出；``None`` 时**不联网**（调用方负责传入）。
    valuation : dict | None
        :func:`fetch_valuation` 的产出。
    price : pd.DataFrame | pd.Series | float | None
        日线（用于区间涨跌幅等辅助判断）或当前价。
    name : str
        股票名称（用于周期股等判断，可空）。
    news : list | None
        可选。``[{"title","date"}...]``，用于识别订单/重组类催化剂
        （与 :func:`earnings_catalyst` 共用；默认 ``None`` 表示不启用新闻驱动）。

    Returns
    -------
    list[Factor]
        量化**从不抛异常**：数据全空时返回一批低可信度的中性因子。
    """
    fs: list = []
    try:
        frame = frame if isinstance(frame, pd.DataFrame) else None
        if frame is not None and frame.empty:
            frame = None
        if frame is None:
            _note("fundamental_factors：未提供财务表，基本面因子退化为中性/低可信度")
        fs.extend(_growth_factors(frame))
        fs.extend(_quality_factors(frame))
        fs.extend(_cashflow_factors(frame))
        fs.extend(_leverage_factors(frame))
        fs.extend(_valuation_factors(frame, valuation, price))
        fs.extend(_sustainability_factors(frame, name))
        fs.extend(_event_factors(frame, news))
    except Exception as exc:  # noqa: BLE001
        _note("fundamental_factors 内部异常（已降级）：%r" % exc)
        try:
            fs.append(_neutral("fundamental_growth", "业绩成长性",
                               "基本面因子生成异常，已降级", 0.1))
        except Exception:  # noqa: BLE001
            pass
    return fs


# --------------------------------------------------------------------------- #
# 自测
# --------------------------------------------------------------------------- #
def _fake_valuation() -> dict:
    """自测用合成估值字典。"""
    return {"pe_ttm": 22.0, "pe_static": 24.0, "pb": 4.1, "ps": 6.2, "pcf": 20.0,
            "market_cap": 1.5e11, "float_market_cap": 1.4e11, "total_shares": 1.25e9,
            "float_shares": 1.2e9, "dividend_yield": 2.5, "ev_ebitda": float("nan"),
            "pe_percentile": 0.32, "pb_percentile": 0.45, "pe_history_years": 5.0,
            "percentile_method": "price_per_eps_approx", "source": "selftest",
            "industry_pe_median": 26.0, "industry_name": "白酒Ⅱ"}


def _self_test() -> None:
    """离线自测：全部使用 core.selftest_data 合成数据，**不发网络请求**。"""
    import core.selftest_data as sd

    print("=== fundamental self-test ===")
    fin = sd.make_financials_frame(n_periods=14, seed=21)
    # 补齐 fetch_financials 的派生列（模拟真实产出）
    for c in ("报告期类型", "单季营收", "单季净利", "营收环比", "净利环比", "环比增速",
              "研发费用率", "扣非占比", "经营现金流净利比", "扣非净利润", "扣非净利同比",
              "摊薄ROE", "经营现金流同比", "每股净资产", "每股经营现金流",
              "存货周转率", "应收账款周转率", "流动比率", "速动比率", "营业利润",
              "利润总额", "研发费用"):
        if c not in fin.columns:
            fin[c] = np.nan
    periods = ["%d-%s" % (2023 + i // 4, ["03-31", "06-30", "09-30", "12-31"][i % 4])
               for i in range(len(fin))]
    fin["报告期"] = periods
    fin["报告期类型"] = [_period_type(p) for p in periods]
    fin["单季营收"] = fin["营业收入"] / 3.0
    fin["单季净利"] = fin["净利润"] / 3.0
    fin["营收环比"] = fin["营业收入"].pct_change() * 100
    fin["净利环比"] = fin["净利润"].pct_change() * 100
    fin["环比增速"] = fin["净利环比"]
    fin["扣非净利润"] = fin["净利润"] * 0.95
    fin["扣非占比"] = 95.0
    fin["经营现金流净利比"] = fin["经营现金流"] / fin["净利润"]
    fin["研发费用"] = fin["营业收入"] * 0.03
    fin["研发费用率"] = 3.0
    fin["流动比率"] = 2.0
    fin["速动比率"] = 1.5
    fin["每股净资产"] = 20.0
    fin["每股经营现金流"] = fin["每股收益"] * 1.1
    # 合成 EPS 改成平滑正值序列，便于演示 PE 分位（真实数据里 EPS 与股价同量级）
    fin["每股收益"] = 1.20 + 0.08 * np.arange(len(fin))

    ohlcv = sd.make_ohlcv(n=400, seed=42)
    val = _fake_valuation()
    news = [{"title": "公司公告：中标 12 亿元大单，新产品量产", "date": "2026-08-01"},
            {"title": "关于筹划重大资产重组暨关联交易的公告", "date": "2026-08-20"}]

    # 1) 财务表解析 / 单季 / 环比
    f2 = _finalize_financials(fin)
    assert "环比增速" in f2.columns and len(f2) == len(fin)
    print("财务表：%s 列，最新报告期 %s（%s）"
          % (f2.shape, f2["报告期"].iloc[-1], f2["报告期类型"].iloc[-1]))

    # 2) 评分 / 分位 / 事件
    gq = growth_quality_score(f2)
    print("成长质量评分：", gq["score"], "| 分项：growth=%.0f quality=%.0f cashflow=%.0f safety=%.0f"
          % (gq["growth"], gq["quality"], gq["cashflow"], gq["safety"]))
    for line in gq["detail"]:
        print("   -", line)
    vp = valuation_percentile(ohlcv, f2, method="pe")
    print("PE 分位（近似法）：%.3f（样本 %d，窗口 %.2f 年，当前 PE %.1f）"
          % (vp["percentile"], vp["samples"], vp["years"], vp["current"]))
    assert vp["samples"] > 0 and 0.0 <= vp["percentile"] <= 1.0
    ev = earnings_catalyst(f2, news)
    print("催化剂事件 %d 条：%s" % (len(ev), [e["type"] for e in ev]))

    # 3) 因子生成（完整数据）
    fs = fundamental_factors("600519", frame=f2, valuation=val, price=ohlcv,
                             name="贵州茅台", news=news)
    ids = sorted({f.id for f in fs})
    print("因子数量：%d" % len(fs))
    print("因子 id 清单（%d）：%s" % (len(ids), ids))
    assert "fundamental_growth" in ids and "valuation_percentile" in ids
    assert "restructure_event" in ids, "新闻里的重组消息应产出 restructure_event"
    assert all(isinstance(f, Factor) for f in fs)

    # 4) 数据全空用例（不得抛异常）
    empty = fundamental_factors("600519")
    print("空数据因子数量：%d，id：%s" % (len(empty), sorted({f.id for f in empty})))
    assert all(f.reliability <= 0.35 for f in empty)

    # 5) 亏损 + 周期股用例
    loss = f2.copy()
    loss.loc[loss.index[-1], "净利润"] = -1.0e8
    loss.loc[loss.index[-1], "净利同比"] = -180.0
    loss.loc[loss.index[-1], "扣非占比"] = -90.0
    fl = fundamental_factors("600519", frame=loss, valuation=val, price=ohlcv)
    assert any(f.id == "loss_warning" and f.direction < 0 for f in fl)
    print("亏损用例：loss_warning 已触发，因子数 %d" % len(fl))

    print("financials 列（前 12）：", list(FIN_COLUMNS[:12]))
    print("fundamental self-test OK")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
