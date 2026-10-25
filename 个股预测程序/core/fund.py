# -*- coding: utf-8 -*-
"""
场内基金（ETF / LOF / 封闭式基金 / REITs）专用分析。

为什么基金不能用股票那一套
--------------------------
股票看的是"公司赚钱能力 + 资金炒作"，基金看的是**另一组东西**：

    股票：主力资金、龙虎榜、股东户数、PE/PB、业绩预告…
    基金：① **折溢价率**（场内价格 vs 基金净值，谁能想到同一只基金有两个价）
          ② **跟踪误差 / 跟踪偏离**（指数基金的本职工作做得好不好）
          ③ **份额变化**（申购=资金真金白银进来，赎回=流出，比"主力净流入"更真实）
          ④ **规模与流动性**（规模太小有清盘风险，流动性差买卖要吃亏）
          ⑤ **持仓结构**（穿透到底层资产，看它到底买了什么）
          ⑥ **管理人/费率/成立时间**（长期持有的成本与稳定性）

所以本模块与股票模块并存：
* **技术面/大盘联动/消息面** 与股票**共用**（基金也有 K 线、也受大盘影响）；
* **基本面/主力资金** 换成基金的**专属维度**（折溢价、跟踪、份额、规模、持仓）；
* 额外增加 **跟踪指数联动** —— 指数基金涨跌几乎完全由标的指数决定，
  预测它必须先把标的指数看清楚。

数据来源与实测
--------------
* 行情（价格/成交量/换手）：腾讯 ``qt.gtimg.cn``（ETF/LOF/REIT 全部支持）
* 净值历史（单位净值/累计净值/日增长率）：东财天天基金 ``api.fund.eastmoney.com/f10/lsjz``
  —— **折溢价率与跟踪误差的基础**，实测可用
* 基金规模/跟踪标的/申赎状态：天天基金 f10（HTML，正则提取）
* 份额历史：东财 push2（本机网络常被重置，取不到时如实标注）

关键实测样例（2026-10-14，华泰柏瑞沪深300ETF）：
    单位净值 4.2890 ｜ 场内现价 4.294 → **溢价 +0.12%**
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .contracts import (Category, DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Factor,
                        safe_float)
from .utils import get_logger

log = get_logger()

# --------------------------------------------------------------------------- #
# 一、基金类型识别
# --------------------------------------------------------------------------- #
#: 代码前缀 → (类型, 说明, 默认分析侧重)
_FUND_PREFIX: list[tuple[str, str, str]] = [
    # ---- ETF ----
    ("510", "ETF", "沪市ETF（多为宽基/行业指数）"),
    ("511", "ETF", "沪市ETF（货币/债券类）"),
    ("512", "ETF", "沪市ETF（行业主题）"),
    ("513", "ETF", "沪市ETF（跨境QDII）"),
    ("515", "ETF", "沪市ETF（行业主题）"),
    ("516", "ETF", "沪市ETF（行业主题）"),
    ("517", "ETF", "沪市ETF（行业主题）"),
    ("518", "ETF", "沪市ETF（商品/黄金）"),
    ("560", "ETF", "沪市ETF（新代码段）"),
    ("561", "ETF", "沪市ETF（新代码段）"),
    ("562", "ETF", "沪市ETF（新代码段）"),
    ("563", "ETF", "沪市ETF（新代码段）"),
    ("588", "ETF", "沪市ETF（科创/科创板）"),
    ("589", "ETF", "沪市ETF（科创）"),
    ("159", "ETF", "深市ETF"),
    ("16", "LOF", "深市LOF（上市开放式基金）"),
    ("501", "LOF", "沪市LOF"),
    ("502", "LOF", "沪市LOF（分级/创新）"),
    ("506", "LOF", "沪市LOF"),
    ("508", "REITs", "基础设施公募REITs"),
    ("180", "REITs", "基础设施公募REITs（深市）"),
    ("184", "封闭式基金", "深市封闭式基金"),
    ("500", "封闭式基金", "沪市封闭式基金"),
]

#: 类型 → (中文名, 分析侧重)
FUND_TYPE_LABEL = {
    "ETF": "交易型开放式指数基金（ETF）",
    "LOF": "上市开放式基金（LOF）",
    "封闭式基金": "封闭式基金",
    "REITs": "基础设施公募REITs",
}

#: 各类型的分析侧重说明（用于界面提示，体现"按性质特点选指标"）
FUND_FOCUS = {
    "ETF": (
        "ETF 是**被动跟踪指数**的工具，涨跌主要由标的指数决定，因此分析重点是："
        "① 跟踪误差与偏离（它跟踪得好不好）；② 折溢价率（场内价格相对净值贵还是便宜，"
        "溢价过高有回落风险）；③ 份额变化（申购/赎回代表真实资金进出）；"
        "④ 规模与流动性（规模小有清盘风险、流动性差买卖吃亏）；"
        "⑤ 标的指数本身的位置与趋势（决定方向）。"
        "个股那套 PE/PB、龙虎榜、股东户数对它基本不适用。"
    ),
    "LOF": (
        "LOF 既能场内交易也能场外申赎，**折溢价是核心**：溢价时场外申购转托管到场内卖出可套利，"
        "折价时反之。分析重点是折溢价率、成交量（LOF 流动性普遍较差）、"
        "以及主动管理型 LOF 的**净值表现与选股能力**（此时要看基金经理而非指数）。"
    ),
    "封闭式基金": (
        "封闭式基金份额固定、不能申赎，因此**长期折价是常态**。"
        "分析重点是折价率（到期日越近折价越可能收敛）、剩余期限、"
        "以及底层资产的真实价值。折价率是这类基金独有的、最有价值的指标。"
    ),
    "REITs": (
        "REITs 的收益来自**底层不动产的租金现金流 + 资产增值**，"
        "不是靠股价炒作。分析重点是：① 现金分派率（相当于股息率）；"
        "② 折溢价率（相对基金净值）；③ 底层资产类型与区域（产业园/仓储/高速/保租房）；"
        "④ 利率环境（利率下行利好 REITs）。技术上波动小、看长期价值。"
    ),
}


def _prefixes() -> list[tuple[str, str, str]]:
    return sorted(_FUND_PREFIX, key=lambda x: -len(x[0]))


def classify_fund(code: str, name: str = "") -> dict:
    """判断代码/名称是否场内基金，并给出类型与侧重。

    Returns
    -------
    dict：``{"is_fund","type","type_label","focus","reason"}``
    """
    code = re.sub(r"\D", "", str(code or ""))
    nm = str(name or "")
    out = {"is_fund": False, "type": "", "type_label": "", "focus": "",
           "reason": ""}
    if not code:
        return out
    # 名称优先（最可靠）：很多 ETF 名称里直接带 ETF/LOF/REIT
    if "REIT" in nm.upper():
        out.update(is_fund=True, type="REITs", reason="名称含 REIT")
    elif nm.endswith("ETF") or "ETF" in nm:
        out.update(is_fund=True, type="ETF", reason="名称含 ETF")
    elif "LOF" in nm.upper():
        out.update(is_fund=True, type="LOF", reason="名称含 LOF")
    elif "封闭" in nm:
        out.update(is_fund=True, type="封闭式基金", reason="名称含 封闭")
    if not out["is_fund"]:
        for pre, typ, why in _prefixes():
            if code.startswith(pre):
                out.update(is_fund=True, type=typ, reason="代码前缀 %s（%s）" % (pre, why))
                break
    if out["is_fund"]:
        out["type_label"] = FUND_TYPE_LABEL.get(out["type"], out["type"])
        out["focus"] = FUND_FOCUS.get(out["type"], "")
    return out


# --------------------------------------------------------------------------- #
# 二、净值历史（折溢价与跟踪误差的基础）
# --------------------------------------------------------------------------- #
def fetch_nav_history(code: str, pages: int = 3, page_size: int = 200) -> pd.DataFrame:
    """取基金历史净值（东财天天基金）。

    注意：``page_size`` 不要设得过大——实测该接口对 pageSize=200 会返回 20 条，
    而 pageSize=20 正常返回 20 条。所以这里默认按 20 条一页、多翻几页来累积，
    确保有足够样本计算折溢价序列与跟踪误差。
    """
    from .datasource import http_get
    import json as _json

    code = re.sub(r"\D", "", str(code or ""))
    if not code:
        return pd.DataFrame()
    page_size = 20                     # 实测：该接口稳定返回 20 条/页
    pages = max(int(pages or 3), 1)
    rows: list = []
    seen: set = set()
    for page in range(1, pages + 1):
        url = ("https://api.fund.eastmoney.com/f10/lsjz?fundCode=%s&pageIndex=%d&pageSize=%d"
               % (code, page, page_size))
        try:
            txt = http_get(url, timeout=12, retries=1,
                           headers={"Referer": "https://fund.eastmoney.com/"})
            if not txt:
                break
            data = (_json.loads(txt).get("Data") or {}).get("LSJZList") or []
        except Exception as exc:  # noqa: BLE001
            log.debug("净值历史取数失败 %s: %s", code, exc)
            break
        if not data:
            break
        fresh = 0
        for r in data:
            d0 = str(r.get("FSRQ") or "")
            if not d0 or d0 in seen:
                continue
            seen.add(d0)
            fresh += 1
            rows.append({
                "date": d0, "nav": safe_float(r.get("DWJZ")),
                "acc_nav": safe_float(r.get("LJJZ")),
                "nav_pct": safe_float(r.get("JZZZL")),
                "buy_state": r.get("SGZT") or "", "sell_state": r.get("SHZT") or "",
            })
        if fresh == 0:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def nav_metrics(nav: pd.DataFrame, price: pd.DataFrame | None = None) -> dict:
    """基于净值序列计算基金专属指标（含折溢价率）。

    Returns
    -------
    dict：``latest_nav / premium_pct / tracking_* / sharpe / max_dd / …``
    """
    out: dict = {"detail": ""}
    if nav is None or not len(nav):
        out["detail"] = "未取到基金净值（折溢价、跟踪误差无法计算）"
        return out
    d = nav.dropna(subset=["nav"]).copy()
    if len(d) < 3:
        out["detail"] = "净值样本不足"
        return out
    nv = d["nav"].astype(float)
    out["latest_nav"] = round(float(nv.iloc[-1]), 4)
    out["latest_acc_nav"] = round(float(d["acc_nav"].iloc[-1]), 4) \
        if d["acc_nav"].notna().any() else None
    out["nav_date"] = str(d["date"].iloc[-1])[:10]
    out["nav_pct_last"] = round(float(d["nav_pct"].iloc[-1]), 3) \
        if d["nav_pct"].notna().any() else None

    # ---- 折溢价率：必须把"场内收盘价"与"同日净值"对齐 ---- #
    if price is not None and len(price) and "close" in price.columns:
        p = price.copy()
        p["date"] = pd.to_datetime(p["date"], errors="coerce")
        mg = pd.merge(p[["date", "close"]], d[["date", "nav"]], on="date", how="inner")
        mg = mg.dropna()
        if len(mg) >= 5:
            mg["prem"] = (mg["close"] / mg["nav"] - 1) * 100
            prem = mg["prem"].astype(float)
            out["premium_pct"] = round(float(prem.iloc[-1]), 3)
            out["premium_mean_20"] = round(float(prem.tail(20).mean()), 3)
            out["premium_std_20"] = round(float(prem.tail(20).std(ddof=0)), 3)
            out["premium_percentile"] = round(
                float((prem <= prem.iloc[-1]).mean()) * 100, 1)
            out["premium_samples"] = int(len(mg))
            out["premium_min_60"] = round(float(prem.tail(60).min()), 3)
            out["premium_max_60"] = round(float(prem.tail(60).max()), 3)
            out["premium_series"] = [round(float(x), 3) for x in prem.tail(60)]
        else:
            out["detail"] = "净值与收盘价重叠样本不足，折溢价率未计算"

    # ---- 收益与风险（基金口径：用净值而非价格） ---- #
    ret = nv.pct_change().dropna()
    if len(ret) > 20:
        for bars, lab in ((5, "5日"), (20, "20日"), (60, "60日"), (120, "120日")):
            if len(nv) > bars:
                out["nav_ret_%s" % lab] = round(float(nv.iloc[-1] / nv.iloc[-1 - bars] - 1) * 100, 2)
        ann = float(ret.tail(250).std(ddof=0) * math.sqrt(250) * 100) if len(ret) >= 60 else float("nan")
        out["nav_vol_annual"] = round(ann, 2) if ann == ann else None
        # 最大回撤
        cum = (1 + ret).cumprod()
        peak = cum.cummax()
        out["nav_max_dd"] = round(float(((cum / peak) - 1).min()) * 100, 2)
        # 夏普（无风险利率按 2% 近似，可在 config 覆盖）
        try:
            from . import config as _cfg
            rf = float((_cfg.FUND or {}).get("risk_free_rate", 0.02))
        except Exception:  # noqa: BLE001
            rf = 0.02
        if ann == ann and ann > 1e-6:
            tot = float(nv.iloc[-1] / nv.iloc[0] - 1)
            yrs = max(len(nv) / 250.0, 1e-6)
            cagr = (1 + tot) ** (1 / yrs) - 1
            out["nav_cagr"] = round(cagr * 100, 2)
            out["nav_sharpe"] = round((cagr - rf) / (ann / 100.0), 3)
    return out


def tracking_metrics(nav: pd.DataFrame, index_df: pd.DataFrame | None,
                     window: int = 60) -> dict:
    """跟踪误差 / 跟踪偏离 / 信息比率（指数基金的本职工作）。

    跟踪误差 = 日跟踪偏离度的年化标准差；跟踪偏离 = 基金收益 − 指数收益。
    指数基金这两个值越小越好；增强型与主动型基金则看信息比率是否为正。
    """
    out: dict = {"detail": ""}
    if nav is None or not len(nav) or index_df is None or not len(index_df):
        out["detail"] = "缺少净值或标的指数数据，跟踪指标未计算"
        return out
    a = nav[["date", "nav"]].copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce")
    b = index_df[["date", "close"]].copy()
    b["date"] = pd.to_datetime(b["date"], errors="coerce")
    mg = pd.merge(a, b, on="date", how="inner").dropna()
    if len(mg) < 20:
        out["detail"] = "净值与指数重叠样本不足（%d 天），跟踪指标未计算" % len(mg)
        return out
    fr = mg["nav"].astype(float).pct_change()
    ir = mg["close"].astype(float).pct_change()
    diff = (fr - ir).dropna()
    if not len(diff):
        out["detail"] = "跟踪偏离计算失败"
        return out
    out["trk_samples"] = int(len(diff))
    out["trk_daily_mean"] = round(float(diff.mean()) * 100, 4)
    out["trk_annual_te"] = round(float(diff.tail(250).std(ddof=0) * math.sqrt(250) * 100), 3)
    w = min(window, len(diff))
    out["trk_te_%dd" % w] = round(float(diff.tail(w).std(ddof=0) * math.sqrt(250) * 100), 3)
    out["trk_cum_diff"] = round(float(((1 + fr).cumprod().iloc[-1]
                                       / (1 + ir).cumprod().iloc[-1] - 1) * 100), 2)
    # 相关性（指数基金应接近 1）
    both = pd.concat([fr, ir], axis=1).dropna()
    if len(both) >= 20:
        try:
            out["trk_corr"] = round(float(both.iloc[:, 0].corr(both.iloc[:, 1])), 4)
            out["trk_beta"] = round(float(
                both.iloc[:, 0].cov(both.iloc[:, 1]) / both.iloc[:, 1].var()), 4)
        except Exception:  # noqa: BLE001
            pass
    # 信息比率：超额收益 / 跟踪误差
    te = out.get("trk_annual_te") or 0.0
    if te > 1e-6:
        out["trk_info_ratio"] = round(out["trk_cum_diff"] / te, 3)
    return out


# --------------------------------------------------------------------------- #
# 三、基金概况 / 规模 / 跟踪标的
# --------------------------------------------------------------------------- #
def fetch_fund_profile(code: str) -> dict:
    """抓取基金规模、跟踪标的、成立日期、申赎状态（天天基金 f10）。"""
    from .datasource import http_get

    code = re.sub(r"\D", "", str(code or ""))
    out: dict = {}
    if not code:
        return out
    # 基金概况页：跟踪标的、业绩比较基准、成立日期、基金公司
    try:
        html = http_get("https://fundf10.eastmoney.com/jbgk_%s.html" % code,
                        timeout=14, retries=1,
                        headers={"Referer": "https://fund.eastmoney.com/"})
        if html:
            txt = re.sub(r"<[^>]+>", " ", html)
            txt = re.sub(r"\s+", " ", txt)

            def _grab(label: str, n: int = 60) -> str:
                m = re.search(re.escape(label) + r"[：:]?\s*([^：:]{2,%d})" % n, txt)
                return m.group(1).strip() if m else ""

            for key, lab, n in (("tracking_index", "跟踪标的", 40),
                                ("benchmark", "业绩比较基准", 60),
                                ("inception", "成立日期", 20),
                                ("manager", "基金管理人", 40),
                                ("fund_type", "基金类型", 24),
                                ("custodian", "基金托管人", 40)):
                v = _grab(lab, n)
                if v:
                    out[key] = v
    except Exception as exc:  # noqa: BLE001
        log.debug("基金概况抓取失败 %s: %s", code, exc)
    # 规模页：最新规模、份额
    try:
        html = http_get("https://fundf10.eastmoney.com/gmbd_%s.html" % code,
                        timeout=14, retries=1,
                        headers={"Referer": "https://fund.eastmoney.com/"})
        if html:
            txt = re.sub(r"<[^>]+>", " ", html)
            txt = re.sub(r"\s+", " ", txt)
            m = re.search(r"基金规模[：:]?\s*([\d.]+)\s*(亿元|万元)", txt)
            if m:
                val = safe_float(m.group(1))
                out["fund_size"] = val * 1e8 if m.group(2) == "亿元" else val * 1e4
                out["fund_size_text"] = "%s%s" % (m.group(1), m.group(2))
            m2 = re.search(r"(\d{4}-\d{2}-\d{2})\s*[·\s]\s*([\d.]+)\s*亿元", txt)
            if m2:
                out["fund_size_date"] = m2.group(1)
    except Exception as exc:  # noqa: BLE001
        log.debug("基金规模抓取失败 %s: %s", code, exc)
    return out


# --------------------------------------------------------------------------- #
# 四、基金因子
# --------------------------------------------------------------------------- #
def _f(fid, name, direction, strength, reliability, value, detail, theory,
       horizon=None) -> Factor:
    return Factor(fid, name, Category.FUNDSPEC, direction, strength, reliability,
                  value=value, detail=detail, theory=theory,
                  horizon=horizon or ("d1", "d5", "d10", "d20"))


def fund_factors(code: str, name: str, price: pd.DataFrame,
                 nav: pd.DataFrame | None = None,
                 index_df: pd.DataFrame | None = None,
                 profile: dict | None = None,
                 snapshot: dict | None = None) -> list[Factor]:
    """生成场内基金专属因子。

    覆盖六类：折溢价、跟踪质量、净值表现、规模流动性、申赎状态、持仓穿透。
    """
    out: list[Factor] = []
    cls = classify_fund(code, name)
    if not cls["is_fund"]:
        return out
    ftype = cls["type"]
    navm = nav_metrics(nav, price)
    prof = profile or {}
    snap = snapshot or {}

    # ---- ① 折溢价率（基金特有的第一指标）---- #
    prem = navm.get("premium_pct")
    if prem is not None:
        pct = navm.get("premium_percentile")
        strength = min(abs(prem) / 1.5, 0.85)
        if prem > 0.35:
            out.append(_f(
                "fund_premium_high", "基金溢价（场内比净值贵 %.2f%%）" % prem,
                DIR_BEAR, strength, 0.66,
                _pct(prem), 
                "场内价格高于基金净值 %.2f%%（近60日区间 %.2f%%~%.2f%%，当前处于历史 %.0f%% 分位）。"
                "溢价意味着你在为同样的资产多付钱：溢价终会向净值收敛，"
                "买入即承担回落风险；但对已持有者，高溢价也可能反映资金抢筹。"
                "套利资金会通过申购→卖出压低溢价。"
                % (prem, navm.get("premium_min_60") or 0, navm.get("premium_max_60") or 0,
                   pct if pct is not None else 50),
                "ETF/LOF 折溢价与套利机制"))
        elif prem < -0.35:
            out.append(_f(
                "fund_discount", "基金折价（场内比净值便宜 %.2f%%）" % abs(prem),
                DIR_BULL, strength, 0.66, _pct(prem),
                "场内价格低于基金净值 %.2f%%（近60日区间 %.2f%%~%.2f%%，历史 %.0f%% 分位）。"
                "折价相当于折价买入一篮子资产，安全边际较高；"
                "封闭式基金的长期折价还可能随到期日临近而收敛（额外收益来源）。"
                % (abs(prem), navm.get("premium_min_60") or 0, navm.get("premium_max_60") or 0,
                   pct if pct is not None else 50),
                "ETF/LOF 折溢价与套利机制"))
        else:
            out.append(_f(
                "fund_premium_normal", "折溢价正常（%.2f%%）" % prem, DIR_NEUTRAL,
                0.2, 0.6, _pct(prem),
                "场内价格与基金净值基本一致（偏离 %.2f%%），定价有效、无套利空间，"
                "此时基金涨跌基本等于其净值涨跌。" % prem,
                "ETF/LOF 折溢价与套利机制"))
        if navm.get("premium_std_20") is not None:
            sd = navm["premium_std_20"]
            if sd > 0.5:
                out.append(_f(
                    "fund_premium_volatile", "折溢价波动大（20日标准差 %.2f%%）" % sd,
                    DIR_NEUTRAL, min(sd / 3.0, 0.6), 0.58, _pct(sd),
                    "折溢价率波动较大，说明该基金定价效率偏低、套利不顺畅"
                    "（常见于跨境QDII、额度受限或流动性差的品种），"
                    "买卖时点对收益的影响会被放大。", "折溢价波动/定价效率",
                    ("d5", "d10", "d20")))

    # ---- ② 跟踪质量（指数基金的本职工作）---- #
    trk = tracking_metrics(nav, index_df) if index_df is not None else {}
    te = trk.get("trk_annual_te")
    if te is not None:
        good = te <= 2.0
        out.append(_f(
            "fund_tracking_error", "跟踪误差 %.2f%%（年化）" % te,
            DIR_NEUTRAL, min(te / 6.0, 0.7), 0.72, "%.3f%%" % te,
            "年化跟踪误差 %.2f%%%s；累计跟踪偏离 %s%%；与标的指数相关性 %s。"
            "%s"
            % (te,
               "（属于优秀的指数跟踪水平）" if good else
               "（偏大，说明跟踪不够紧密）",
               trk.get("trk_cum_diff"), trk.get("trk_corr"),
               "指数基金的跟踪误差越小越好——它衡量基金是否忠实复制了标的指数。" ),
            "指数化投资/跟踪误差理论", ("d5", "d10", "d20", "m1", "q1")))
        ir = trk.get("trk_info_ratio")
        if ir is not None:
            out.append(_f(
                "fund_info_ratio", "信息比率 %.2f" % ir,
                DIR_BULL if ir > 0.3 else (DIR_BEAR if ir < -0.3 else DIR_NEUTRAL),
                min(abs(ir) / 2.0, 0.7), 0.66, "%.3f" % ir,
                "信息比率 = 相对标的指数的超额收益 / 跟踪误差 = %.2f。"
                "%s" % (ir,
                        "持续为正说明该基金能稳定跑赢标的（增强型或管理能力强）；"
                        "持续为负说明长期跑输标的（费率或复制效率拖累）。"
                        if abs(ir) > 0.3 else
                        "接近 0 说明基本贴合标的指数，属标准被动跟踪。"),
                "信息比率/增强型基金评价", ("d20", "m1", "q1", "h1")))
    beta = trk.get("trk_beta")
    corr = trk.get("trk_corr")
    if corr is not None:
        out.append(_f(
            "fund_index_corr", "与标的指数相关性 %.3f" % corr, DIR_NEUTRAL,
            0.25, 0.7, "%.4f" % corr,
            "基金净值与标的指数日收益相关性 %.3f，Beta %.3f。%s"
            % (corr, beta if beta is not None else float("nan"),
               "相关性接近 1，说明该基金几乎完全由标的指数决定，"
               "**预测它本质上就是预测标的指数**。" if corr > 0.95 else
               "相关性偏低，说明基金并非纯被动跟踪（可能是主动型或抽样复制），"
               "需要额外考察管理人的选股能力。"),
            "资本资产定价模型（CAPM）", ("d5", "d10", "d20", "m1")))

    # ---- ③ 净值表现与风险（基金口径）---- #
    if navm.get("nav_max_dd") is not None:
        mdd = navm["nav_max_dd"]
        out.append(_f(
            "fund_nav_risk", "净值最大回撤 %.2f%%" % mdd, DIR_NEUTRAL,
            0.2, 0.7, "%.2f%%" % mdd,
            "历史净值最大回撤 %.2f%%；年化波动 %s%%；夏普比率 %s；年化收益 %s%%。"
            "基金的评价应以**净值**为准（而不是场内价格的短期波动），"
            "回撤与波动决定了持有体验和仓位是否合适。"
            % (mdd, navm.get("nav_vol_annual"), navm.get("nav_sharpe"),
               navm.get("nav_cagr")),
            "基金绩效评价（夏普/最大回撤）", ("m1", "q1", "h1", "y1")))
    shp = navm.get("nav_sharpe")
    if shp is not None:
        # 注意：这段文案里既有要转义的百分号、又有格式占位符，用 % 格式化极易出错
        # （曾报 ValueError: unsupported format character），改用 str.format 更稳。
        if shp > 1.0:
            _comment = "风险调整后收益优秀。"
        elif shp > 0:
            _comment = "风险调整后收益一般。"
        else:
            _comment = "风险调整后收益为负 —— 承担了波动却没换来收益。"
        out.append(_f(
            "fund_sharpe", "夏普比率 %.2f" % shp,
            DIR_BULL if shp > 1.0 else (DIR_BEAR if shp < 0 else DIR_NEUTRAL),
            min(abs(shp) / 2.0, 0.6), 0.62, "%.3f" % shp,
            "近一年年化收益 {}%、年化波动 {}% → 夏普比率 {:.2f}"
            "（无风险利率按 2% 计）。{}".format(
                navm.get("nav_cagr"), navm.get("nav_vol_annual"), shp, _comment),
            "夏普比率", ("q1", "h1", "y1")))

    # ---- ④ 规模与流动性 ----
    size = prof.get("fund_size")
    if size is not None:
        size_yi = size / 1e8
        if size_yi < 0.5:
            out.append(_f(
                "fund_size_risk", "规模偏小（%.2f 亿）有清盘风险" % size_yi,
                DIR_BEAR, 0.62, 0.72, "%.2f 亿" % size_yi,
                "基金规模仅 %.2f 亿元。规模长期低于 5000 万元会触发**清盘**风险，"
                "且规模小的基金买卖冲击成本高、折溢价容易失真。" % size_yi,
                "基金清盘规则/流动性风险", ("d20", "m1", "q1", "h1")))
        elif size_yi >= 50:
            out.append(_f(
                "fund_size_large", "规模充裕（%.2f 亿）" % size_yi, DIR_BULL,
                0.35, 0.7, "%.2f 亿" % size_yi,
                "基金规模 %.2f 亿元，流动性好、清盘风险极低，"
                "适合作为中长期配置工具。" % size_yi,
                "基金规模与流动性", ("m1", "q1", "h1", "y1")))
    turn = safe_float(snap.get("turnover"))
    if turn == turn:
        out.append(_f(
            "fund_liquidity", "场内换手率 %.2f%%" % turn, DIR_NEUTRAL,
            0.25 if turn < 0.5 else 0.15, 0.66, "%.2f%%" % turn,
            "场内换手率 %.2f%%。%s"
            % (turn,
               "换手偏低说明流动性一般，大额买卖可能明显影响成交价，"
               "建议用限价单并分批操作。" if turn < 0.5 else
               "流动性尚可，正常买卖不会明显冲击价格。"),
            "市场微观结构/流动性", ("d1", "d5")))

    # ---- ⑤ 申赎状态（LOF/场外份额）---- #
    if nav is not None and len(nav):
        bs = str(nav["buy_state"].iloc[-1] or "")
        ss = str(nav["sell_state"].iloc[-1] or "")
        if bs or ss:
            limited = ("暂停" in bs or "限制" in bs or "暂停" in ss or "限制" in ss)
            if limited:
                out.append(_f(
                    "fund_purchase_limited", "申购受限（%s）" % (bs or ss),
                    DIR_NEUTRAL, 0.4, 0.7, bs or ss,
                    "当前申购状态：%s；赎回状态：%s。申购受限会**切断套利机制**，"
                    "使场内价格更容易脱离净值（高溢价可能持续），"
                    "常见于跨境QDII（外汇额度用完）与热门主题基金。" % (bs, ss),
                    "申赎机制与折溢价形成", ("d5", "d10", "d20")))

    # ---- ⑥ 持仓穿透（有持仓数据时）---- #
    try:
        hold = fetch_fund_holdings(code)
        if hold:
            top = hold[:5]
            names = "、".join("%s(%.1f%%)" % (h.get("name"), (h.get("ratio") or 0))
                              for h in top if h.get("name"))
            conc = sum((h.get("ratio") or 0) for h in top[:5])
            out.append(_f(
                "fund_holdings", "前五大持仓集中度 %.1f%%" % conc, DIR_NEUTRAL,
                0.2, 0.68, "%.1f%%" % conc,
                "前五大持仓合计 %.1f%%：%s。"
                "基金虽分散，但若前几大持仓占比很高，"
                "其涨跌仍会被少数标的左右。" % (conc, names or "—"),
                "组合集中度/穿透持仓", ("m1", "q1")))
    except Exception as exc:  # noqa: BLE001
        log.debug("基金持仓获取失败 %s: %s", code, exc)

    # ---- ⑦ REITs 专属：现金分派率 ---- #
    if ftype == "REITs":
        m = re.search(r"分派率[：:]?\s*([\d.]+)\s*%", str(prof.get("benchmark") or ""))
        if m:
            out.append(_f(
                "fund_reit_yield", "现金分派率 %s%%" % m.group(1), DIR_BULL,
                min(safe_float(m.group(1)) / 8.0, 0.6), 0.6, m.group(1) + "%",
                "REITs 的核心收益来源是底层资产租金产生的**现金分派**，"
                "分派率 %s%% 相当于该类资产的股息率，是长期持有的主要回报。" % m.group(1),
                "REITs 估值（现金分派率）", ("q1", "h1", "y1")))
    return out


def _pct(v) -> str:
    try:
        f = float(v)
        return "—" if f != f else "%+.2f%%" % f
    except (TypeError, ValueError):
        return "—"


# --------------------------------------------------------------------------- #
# 五、持仓穿透
# --------------------------------------------------------------------------- #
def fetch_fund_holdings(code: str, top: int = 10) -> list:
    """抓取基金前十大持仓（天天基金 f10 接口）。"""
    from .datasource import http_get
    import json as _json

    code = re.sub(r"\D", "", str(code or ""))
    if not code:
        return []
    url = ("https://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=jjcc"
           "&code=%s&topline=%d" % (code, top))
    try:
        txt = http_get(url, timeout=14, retries=1,
                       headers={"Referer": "https://fund.eastmoney.com/"})
    except Exception as exc:  # noqa: BLE001
        log.debug("持仓接口失败 %s: %s", code, exc)
        return []
    if not txt:
        return []
    # 返回形如：var apidata={ content:"<div…><tbody>…</tbody>",arryear:[…]}
    body = txt
    m = re.search(r'content:"(.*?)",arryear', body, re.S)
    if m:
        body = m.group(1).replace('\\"', '"')
    rows: list = []
    try:
        import pandas as pd  # noqa: F401

        tables = re.findall(r"<tbody>(.*?)</tbody>", body, re.S)
        for tb in tables[:1]:
            for tr in re.findall(r"<tr>(.*?)</tr>", tb, re.S):
                tds = [re.sub(r"<[^>]+>", "", x).strip()
                       for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
                if len(tds) < 3:
                    continue
                # 典型列：序号 股票代码 股票名称 最新价 涨跌幅 相关资讯 占净值比例 持股数 持仓市值
                nums = [t for t in tds if re.fullmatch(r"[\d.]+%?", t or "")]
                ratio = None
                for t in tds:
                    if t.endswith("%"):
                        ratio = safe_float(t.rstrip("%"))
                        break
                name = ""
                for t in tds:
                    if t and not re.fullmatch(r"[\d.%\-]+", t) and len(t) <= 12:
                        name = t
                        break
                if name and ratio is not None:
                    rows.append({"name": name, "ratio": ratio})
    except Exception as exc:  # noqa: BLE001
        log.debug("持仓解析失败 %s: %s", code, exc)
    return rows[:top]



# --------------------------------------------------------------------------- #
# 成分股（持仓）分析
# --------------------------------------------------------------------------- #
def fetch_fund_holdings_detail(code: str, top: int = 10) -> list:
    """抓前十大持仓，带**股票代码**与占净值比例。

    Returns
    -------
    list[dict]：``[{"code","name","ratio"}, …]``，按 ratio 降序
    """
    from .datasource import http_get

    code = re.sub(r"\D", "", str(code or ""))
    if not code:
        return []
    url = ("https://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=jjcc"
           "&code=%s&topline=%d" % (code, max(top, 10)))
    try:
        txt = http_get(url, timeout=14, retries=1,
                       headers={"Referer": "https://fund.eastmoney.com/"})
    except Exception as exc:  # noqa: BLE001
        log.debug("持仓明细接口失败 %s: %s", code, exc)
        return []
    if not txt:
        return []
    body = txt
    m = re.search(r'content:"(.*?)",arryear', txt, re.S)
    if m:
        body = m.group(1).replace('\\"', '"')
    rows: list = []
    seen: set = set()
    for tb in re.findall(r"<tbody>(.*?)</tbody>", body, re.S)[:1]:
        for tr in re.findall(r"<tr>(.*?)</tr>", tb, re.S):
            tds = [re.sub(r"<[^>]+>", "", x).strip()
                   for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            if len(tds) < 7:
                continue
            # 典型列：序号 股票代码 股票名称 … 占净值比例 持股数 持仓市值
            scode = ""
            for t in tds[1:4]:
                if re.fullmatch(r"\d{6}", t or ""):
                    scode = t
                    break
            if not scode or scode in seen:
                continue
            name = ""
            for t in tds[2:5]:
                if t and not re.fullmatch(r"[\d.%\-]+", t) and len(t) <= 14:
                    name = t
                    break
            ratio = None
            for t in tds:
                if t.endswith("%") and re.match(r"^[\d.]+%$", t):
                    ratio = safe_float(t.rstrip("%"))
                    break
            if not name:
                continue
            seen.add(scode)
            rows.append({"code": scode, "name": name,
                         "ratio": ratio if ratio is not None else 0.0})
    rows.sort(key=lambda x: -(x.get("ratio") or 0))
    return rows[:top]


def analyze_holdings(code: str, nav: pd.DataFrame | None = None,
                     top: int = 10, max_workers: int = 6) -> dict:
    """分析基金成分股（持仓）的当前状态。

    做法：抓前十大持仓 → 并发取个股日线 → 按权重算组合强弱 + 广度。

    Returns
    -------
    dict：``{"items","weighted_ret_1d/5d/20d","breadth","concentration",
    "detail","top_heavy"}``
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out: dict = {"items": [], "detail": "", "concentration": {}}
    holds = fetch_fund_holdings_detail(code, top=top)
    if not holds:
        out["detail"] = "未取到前十大持仓（基金可能尚未披露季报）"
        return out

    from .datasource import fetch_daily

    def _one(h: dict):
        try:
            df = fetch_daily(h["code"], days=180)
            return h, df
        except Exception as exc:  # noqa: BLE001
            log.debug("成分股 %s 取数失败: %s", h["code"], exc)
            return h, None

    frames: dict = {}
    with ThreadPoolExecutor(max_workers=max(1, min(len(holds), max_workers))) as pool:
        futs = [pool.submit(_one, h) for h in holds]
        for fu in as_completed(futs):
            try:
                h, df = fu.result()
                if df is not None and len(df) >= 25:
                    frames[h["code"]] = df
            except Exception:  # noqa: BLE001
                continue

    items: list = []
    for h in holds:
        df = frames.get(h["code"])
        row = dict(h)
        if df is not None and len(df) >= 21:
            try:
                c = df["close"].astype(float).dropna()
                row["ret_1d"] = round(float(c.iloc[-1] / c.iloc[-2] - 1) * 100, 2)
                row["ret_5d"] = round(float(c.iloc[-1] / c.iloc[-6] - 1) * 100, 2)
                row["ret_20d"] = round(float(c.iloc[-1] / c.iloc[-21] - 1) * 100, 2)
                row["above_ma20"] = bool(c.iloc[-1] >= c.tail(20).mean())
                win = min(len(c), 250)
                hi, lo = float(c.tail(win).max()), float(c.tail(win).min())
                row["pos_pct"] = round(
                    (float(c.iloc[-1]) - lo) / max(hi - lo, 1e-9) * 100, 1)
            except Exception:  # noqa: BLE001
                pass
        items.append(row)
    out["items"] = items

    # ---- 按权重加权的组合涨跌（只统计有数据的成分股，权重归一）---- #
    def _wavg(field: str):
        tot, acc = 0.0, 0.0
        for it in items:
            v = it.get(field)
            w = float(it.get("ratio") or 0.0)
            if v is None or w <= 0:
                continue
            tot += w
            acc += w * float(v)
        return round(acc / tot, 3) if tot > 0 else None

    out["weighted_ret_1d"] = _wavg("ret_1d")
    out["weighted_ret_5d"] = _wavg("ret_5d")
    out["weighted_ret_20d"] = _wavg("ret_20d")

    # ---- 广度：多少成分股在涨 / 站上 MA20 ---- #
    valid = [it for it in items if it.get("ret_1d") is not None]
    if valid:
        up = sum(1 for it in valid if (it.get("ret_1d") or 0) > 0)
        out["breadth_up_pct"] = round(up / len(valid) * 100, 1)
        out["breadth_valid"] = len(valid)
    ma = [it for it in items if it.get("above_ma20") is not None]
    if ma:
        n = sum(1 for it in ma if it.get("above_ma20"))
        out["breadth_ma20_pct"] = round(n / len(ma) * 100, 1)

    # ---- 集中度 ---- #
    ratios = [float(it.get("ratio") or 0) for it in items]
    out["concentration"] = {
        "top1": round(ratios[0], 2) if ratios else None,
        "top5": round(sum(ratios[:5]), 2),
        "top10": round(sum(ratios[:10]), 2),
        "n": len([r for r in ratios if r > 0]),
    }
    out["top_heavy"] = bool(out["concentration"]["top5"] >= 40)

    # ---- 与基金净值的偏离：估算 vs 实际 ---- #
    if nav is not None and len(nav) and out.get("weighted_ret_1d") is not None:
        try:
            nv = nav["nav"].astype(float).dropna()
            if len(nv) >= 2:
                actual = float(nv.iloc[-1] / nv.iloc[-2] - 1) * 100
                out["nav_ret_1d"] = round(actual, 3)
                # 估算只覆盖前十大，通常是实际波动的一部分，故给区间描述
                out["gap_vs_nav"] = round(out["weighted_ret_1d"] - actual, 3)
        except Exception:  # noqa: BLE001
            pass

    bits = ["前十大持仓合计 %.2f%%" % out["concentration"]["top10"]]
    if out.get("weighted_ret_1d") is not None:
        bits.append("加权涨跌（今日）%+.2f%%" % out["weighted_ret_1d"])
    if out.get("weighted_ret_20d") is not None:
        bits.append("加权近20日 %+.2f%%" % out["weighted_ret_20d"])
    if out.get("breadth_up_pct") is not None:
        bits.append("%d 只成分股中 %.0f%% 上涨" % (out["breadth_valid"],
                                                out["breadth_up_pct"]))
    if out.get("breadth_ma20_pct") is not None:
        bits.append("%.0f%% 站上 MA20" % out["breadth_ma20_pct"])
    if out["top_heavy"]:
        bits.append("**集中度高（前五 %.1f%%），基金涨跌被少数个股主导**"
                    % out["concentration"]["top5"])
    out["detail"] = "；".join(bits)
    return out


def holdings_factors(ana: dict, ftype: str = "") -> list:
    """把成分股分析转成因子。"""
    out: list[Factor] = []
    if not ana or not ana.get("items"):
        return out
    wr1 = ana.get("weighted_ret_1d")
    wr20 = ana.get("weighted_ret_20d")
    if wr1 is not None:
        out.append(_f(
            "fund_holding_strength", "成分股加权涨跌 %+.2f%%" % wr1,
            DIR_BULL if wr1 > 0.2 else (DIR_BEAR if wr1 < -0.2 else DIR_NEUTRAL),
            min(abs(wr1) / 3.0 + 0.2, 0.8), 0.66,
            "%+.2f%%" % wr1,
            "按持仓比例加权，前十大成分股今日平均 %+.2f%%（近20日 %s%%）。"
            "成分股实时交易，其强弱**领先于**基金净值——净值每天只在收盘后更新一次。"
            % (wr1, wr20 if wr20 is not None else "—"),
            "组合穿透/净值预估", ("d1", "d5")))
    bu = ana.get("breadth_up_pct")
    if bu is not None:
        out.append(_f(
            "fund_holding_breadth", "成分股广度 %.0f%%" % bu,
            DIR_BULL if bu >= 60 else (DIR_BEAR if bu <= 30 else DIR_NEUTRAL),
            min(abs(bu - 50) / 100.0 * 1.8 + 0.2, 0.8), 0.64,
            "%d 只中 %.0f%% 上涨" % (ana.get("breadth_valid") or 0, bu),
            "前十大成分股中 %.0f%% 今日上涨，%.0f%% 站上 MA20。"
            "广度高说明是整体走强（非个别拉动），基金跟随概率大。"
            % (bu, ana.get("breadth_ma20_pct") or 0),
            "组合广度/内部一致性", ("d1", "d5", "d10")))
    conc = ana.get("concentration") or {}
    if conc.get("top5"):
        heavy = ana.get("top_heavy")
        out.append(_f(
            "fund_concentration", "持仓集中度 前五 %.1f%%" % conc["top5"],
            DIR_NEUTRAL, min(conc["top5"] / 80.0 + 0.15, 0.7), 0.68,
            "前五 %.1f%%｜前十 %.1f%%" % (conc["top5"], conc.get("top10") or 0),
            "前五大持仓合计 %.1f%%，前十大 %.1f%%。%s"
            % (conc["top5"], conc.get("top10") or 0,
               "**集中度很高**：基金涨跌基本由这几只决定，"
               "等于间接买入这几只股票，需关注它们的风险。" if heavy else
               "分散度尚可，单一标的对基金影响有限。"),
            "组合集中度/穿透持仓", ("m1", "q1")))
    return out


# --------------------------------------------------------------------------- #
# 六、一站式入口
# --------------------------------------------------------------------------- #
def analyze_fund(code: str, name: str, price: pd.DataFrame,
                 index_df: pd.DataFrame | None = None,
                 snapshot: dict | None = None,
                 nav: pd.DataFrame | None = None,
                 profile: dict | None = None) -> dict:
    """场内基金完整分析：类型 → 净值/折溢价 → 跟踪 → 因子。"""
    cls = classify_fund(code, name)
    result: dict = {"classify": cls, "nav": {}, "tracking": {}, "profile": {},
                    "factors": [], "detail": ""}
    if not cls["is_fund"]:
        result["detail"] = "不是场内基金代码，跳过基金专属分析"
        return result
    if nav is None:
        nav = fetch_nav_history(code)
    if profile is None:
        profile = fetch_fund_profile(code)
    navm = nav_metrics(nav, price)
    trk = tracking_metrics(nav, index_df) if index_df is not None else {}
    result["nav"] = navm
    result["tracking"] = trk
    result["profile"] = profile or {}
    result["type"] = cls["type"]
    result["type_label"] = cls["type_label"]
    # ---- 成分股（持仓）分析：基金的"穿透"视角 ----
    hold_ana: dict = {}
    try:
        hold_ana = analyze_holdings(code, nav)
        result["holdings_analysis"] = hold_ana
    except Exception as exc:  # noqa: BLE001
        log.debug("成分股分析失败 %s: %s", code, exc)
    result["factors"] = fund_factors(code, name, price, nav, index_df, profile, snapshot)
    try:
        result["factors"] = list(result["factors"]) + holdings_factors(
            hold_ana, cls.get("type") or "")
    except Exception as exc:  # noqa: BLE001
        log.debug("成分股因子生成失败: %s", exc)
    bits = ["已识别为 %s" % cls["type_label"]]
    if navm.get("premium_pct") is not None:
        bits.append("当前折溢价 %+.2f%%" % navm["premium_pct"])
    if navm.get("latest_nav"):
        bits.append("最新净值 %s（%s）" % (navm["latest_nav"], navm.get("nav_date")))
    if trk.get("trk_annual_te") is not None:
        bits.append("年化跟踪误差 %.2f%%" % trk["trk_annual_te"])
    elif trk.get("detail"):
        bits.append(trk["detail"])
    if profile.get("fund_size_text"):
        bits.append("规模 %s" % profile["fund_size_text"])
    if hold_ana.get("weighted_ret_1d") is not None:
        bits.append("成分股加权 %+.2f%%" % hold_ana["weighted_ret_1d"])
    result["detail"] = "；".join(bits)
    return result


if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")

    print("=== 1. 类型识别 ===")
    cases = [
        ("510300", "沪深300ETF华泰柏瑞", "ETF"),
        ("159915", "创业板ETF易方达", "ETF"),
        ("588000", "科创50ETF华夏", "ETF"),
        ("518880", "黄金ETF华安", "ETF"),
        ("161725", "白酒基金LOF", "LOF"),
        ("501050", "50AHLOF", "LOF"),
        ("508000", "华安张江产业园REIT", "REITs"),
        ("600519", "贵州茅台", None),
        ("000001", "平安银行", None),
    ]
    ok = 0
    for code, nm, want in cases:
        got = classify_fund(code, nm)
        hit = (got["type"] == want) if want else (not got["is_fund"])
        ok += 1 if hit else 0
        print("  %-8s %-20s → %-6s %-14s %s" % (
            code, nm, got["type"] or "（非基金）", got["reason"][:16],
            "✓" if hit else "✗ 期望 %s" % want))
    print("类型识别正确 %d/%d" % (ok, len(cases)))
    assert ok == len(cases), "类型识别有误"

    print("\n=== 2. 净值与折溢价（离线构造）===")
    from .selftest_data import make_ohlcv

    n = 160
    df = make_ohlcv(n, seed=11, start_price=4.0)
    nav = pd.DataFrame({
        "date": df["date"],
        "nav": df["close"] * 0.998,          # 制造约 +0.2% 溢价
        "acc_nav": df["close"],
        "nav_pct": df["close"].pct_change() * 100,
        "buy_state": ["开放申购"] * n, "sell_state": ["开放赎回"] * n,
    })
    nm = nav_metrics(nav, df)
    print("  最新净值 %.4f，折溢价 %+.3f%%，20日均值 %+.3f%%，分位 %.1f%%"
          % (nm["latest_nav"], nm["premium_pct"], nm["premium_mean_20"],
             nm["premium_percentile"]))
    assert abs(nm["premium_pct"] - 0.2) < 0.06, nm["premium_pct"]

    print("\n=== 3. 跟踪指标（离线构造）===")
    idx = df.copy()
    idx["close"] = df["close"] * 1.0004          # 略强于基金的指数
    trk = tracking_metrics(nav, idx)
    print("  年化跟踪误差 %.3f%%，累计偏离 %.2f%%，相关性 %.4f，Beta %.4f，信息比率 %s"
          % (trk["trk_annual_te"], trk["trk_cum_diff"], trk["trk_corr"],
             trk["trk_beta"], trk.get("trk_info_ratio")))
    assert trk["trk_corr"] > 0.99, trk["trk_corr"]

    print("\n=== 4. 因子生成 ===")
    prof = {"fund_size_text": "1200.00亿元", "fund_size": 1200e8,
            "tracking_index": "沪深300指数"}
    fs = fund_factors("510300", "沪深300ETF华泰柏瑞", df, nav, idx, prof,
                      {"turnover": 3.59})
    print("  生成 %d 条基金专属因子：" % len(fs))
    for f in fs:
        print("     %-34s %s 强度%.2f 可信度%.2f"
              % (f.name[:32], {1: "多", -1: "空", 0: "中"}[f.direction],
                 f.strength, f.reliability))
    assert len(fs) >= 5, "因子数量偏少"
    for f in fs:
        assert f.category == Category.FUNDSPEC
        assert 0 <= f.strength <= 1 and 0 <= f.reliability <= 1
        assert f.detail and f.theory and "None" not in str(f.value)

    print("\n=== 5. 边界情况 ===")
    assert classify_fund("", "")["is_fund"] is False
    assert nav_metrics(None, None)["detail"]
    assert tracking_metrics(None, None)["detail"]
    assert fund_factors("600519", "贵州茅台", df) == []
    print("  空输入/非基金代码均安全")
    print("\nfund self-test OK")
