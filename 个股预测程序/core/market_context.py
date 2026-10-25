# -*- coding: utf-8 -*-
"""
市场环境（宏观/政治/社会/政策）+ 涨跌主要动能对象及其异动意图。

第一部分：市场环境四象限
------------------------
A股是"政策+流动性晴雨表"，所以环境分析必须分层：
  · **政策**：货币政策（LPR/降准降息）、产业政策、监管松紧
  · **经济**：PMI、CPI/PPI、M2（增长与通胀的组合）
  · **流动性**：两融、成交额、北向（已在 market_fundamental，这里引用结论）
  · **外部**：美联储/汇率/地缘（用可用代理，缺则如实标注）

诚实的边界：**政治、社会、地缘这类因素无法可靠自动量化**。
本模块只对"有数据可依"的部分打分，并把无法量化的部分列成
**人工核查清单**（checklist），而不是编造一个数字。

第二部分：涨跌主要动能对象与异动意图
------------------------------------
指数涨跌不等于"所有股票都在涨"，而是**少数权重/主线决定**。
识别"谁在推动"比"涨了多少"更重要：
  · 动能对象：按板块当日涨跌幅×成交额估算对指数的贡献，取前几名
  · 异动意图：结合量能、资金净流入、领涨股、宽度，判定
      启动（放量+资金进+宽度扩散）
      加速（涨幅扩大+量能续增）
      滞涨（涨幅大但资金流出/宽度收缩）
      护盘（权重涨但多数板块跌）
      砸盘（权重跌且放量）
      对倒（成交放大但价格不动）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import Category, DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Factor, safe_float
from .utils import get_logger

log = get_logger()

class _MacroDone(Exception):
    """内部哨兵：已由 macro_source 取到数据，跳过旧的 akshare 路径。"""


#: 宏观指标候选函数名（按优先级尝试，避免依赖单一接口名）
_MACRO_PROBES = {
    "pmi": ("macro_china_pmi", "macro_china_pmi_yearly"),
    "cpi": ("macro_china_cpi_yearly", "macro_china_cpi"),
    "ppi": ("macro_china_ppi_yearly", "macro_china_ppi"),
    "m2": ("macro_china_m2_yearly", "macro_china_m2"),
    "lpr": ("macro_china_lpr",),
}


def _age_days(date_str) -> float | None:
    """数据日期距今多少天；解析失败返回 None。"""
    if not date_str:
        return None
    try:
        d = pd.to_datetime(str(date_str)[:10], errors="coerce")
        if d is None or pd.isna(d):
            return None
        return float((pd.Timestamp.now().normalize() - d).days)
    except Exception:  # noqa: BLE001
        return None


def _probe(fn_names: tuple, **kw):
    """依次尝试 akshare 函数名，返回第一个非空 DataFrame。"""
    try:
        import akshare as ak
    except Exception:  # noqa: BLE001
        return None
    for nm in fn_names:
        f = getattr(ak, nm, None)
        if f is None:
            continue
        try:
            df = f(**kw)
            if df is not None and len(df):
                return df
        except Exception:  # noqa: BLE001
            continue
    return None


def _last_num(df, prefer=("今值", "现值", "value", "最新值"),
              date_keys=("日期", "月份", "统计时间", "时间",
                         # akshare 的 LPR/利率类接口日期列叫 TRADE_DATE
                         "TRADE_DATE", "PUBLISH_DATE", "REPORT_DATE",
                         "date", "month")):
    """从宏观表里取最新一个数值与日期（列名各家不同，按关键词找）。"""
    if df is None or not len(df):
        return None, None
    try:
        d = df.copy()
        dcol = next((c for c in d.columns if any(k in str(c) for k in date_keys)), None)
        if dcol is not None:
            d[dcol] = pd.to_datetime(d[dcol], errors="coerce")
            d = d.dropna(subset=[dcol]).sort_values(dcol)
        vcol = None
        for k in prefer:
            vcol = next((c for c in d.columns if k in str(c)), None)
            if vcol:
                break
        if vcol is None:
            nums = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
            vcol = nums[0] if nums else None
        if vcol is None:
            return None, None
        s = pd.to_numeric(d[vcol], errors="coerce").dropna()
        if not len(s):
            return None, None
        dt = str(d[dcol].iloc[-1])[:10] if dcol is not None else ""
        return float(s.iloc[-1]), dt
    except Exception:  # noqa: BLE001
        return None, None


# --------------------------------------------------------------------------- #
# 一、环境四象限
# --------------------------------------------------------------------------- #
def macro_environment() -> dict:
    """取宏观指标并按四象限打分（政策/经济/流动性/外部）。"""
    out = {"indicators": {}, "policy": {}, "economy": {}, "liquidity": {},
           "external": {}, "checklist": [], "detail": ""}
    # ---- 取数：东财 datacenter 优先 → akshare → 手动输入 ----
    # akshare 的 macro_china_* 上游冻结在 2025-08/09，而东财经济报表是新鲜的
    # （实测 2026-09），所以取数顺序改为 MO.get_macro 的统一链路。
    try:
        from . import macro_source as MO

        got_all = MO.collect()
        for key, rec in (got_all or {}).items():
            if rec.get("value") is None:
                continue
            out["indicators"][key] = {
                "value": rec["value"], "date": rec.get("date") or "",
                "age_days": rec.get("age_days"),
                "source": rec.get("source") or "",
                "stale": not rec.get("fresh"),
                # CPI/PPI 另附当月同比，两个口径都展示（易混点）
                "mom_same": rec.get("mom_same"),
                # 口径说明（统计口径 / 字段与换算 / 备注）
                "caliber": rec.get("caliber") or "",
                "caliber_field": rec.get("caliber_field") or "",
                "caliber_note": rec.get("caliber_note") or "",
            }
            if not rec.get("fresh"):
                out.setdefault("stale", {})[key] = (
                    round(rec.get("age_days") or 0),
                    MACRO_MAX_AGE.get(key, 70))
        if out["indicators"]:
            out["detail_sources"] = {k: v.get("source")
                                     for k, v in out["indicators"].items()}
            raise _MacroDone()
    except _MacroDone:
        pass
    except Exception as exc:  # noqa: BLE001
        log.debug("macro_source 取数失败，回退旧的 akshare 路径: %s", exc)

    # ---- 取数（每项独立容错）+ 新鲜度校验 ----
    # 宏观是月频数据，接口返回的"最新值"可能已经滞后数月。
    # 超过上限的一律标注过期并**不参与打分**（与市场基本面替代物同一原则）。
    stale: dict = {}
    for key, names in _MACRO_PROBES.items():
        # 不覆盖 macro_source 取到的**新鲜**值（否则东财的新数据会被
        # akshare 的旧数据盖掉——上一版的真实 bug）
        _prev = out["indicators"].get(key) or {}
        if _prev.get("value") is not None and not _prev.get("stale"):
            continue
        try:
            df = _probe(names)
            val, dt = _last_num(df)
            if val is None:
                continue
            age = _age_days(dt)
            limit = MACRO_MAX_AGE.get(key, 70)
            rec = {"value": round(val, 2), "date": dt or "", "age_days": age}
            if age is not None and age > limit:
                rec["stale"] = True
                stale[key] = (round(age), limit)
            elif age is None:
                # 取不到日期 → 无法验证新鲜度，按"不可信"处理但不排除，
                # 只标注提示（避免因接口不给日期就丢掉全部指标）
                rec["date_unknown"] = True
            out["indicators"][key] = rec
        except Exception as exc:  # noqa: BLE001
            log.debug("宏观 %s 取数失败: %s", key, exc)
    out["stale"] = {k: v for k, v in stale.items()}

    ind = {k: v for k, v in out["indicators"].items() if not v.get("stale")}
    # ---- 经济象限：PMI 荣枯线、CPI/PPI 通胀组合 ----
    econ_parts = []
    pmi = (ind.get("pmi") or {}).get("value")
    if pmi is not None:
        # PMI 50 为荣枯线；偏离 ±2 达饱和
        v = max(-1.0, min(1.0, (pmi - 50) / 2.0))
        econ_parts.append(("PMI %.1f" % pmi, v, 1.0))
    cpi = (ind.get("cpi") or {}).get("value")
    ppi = (ind.get("ppi") or {}).get("value")
    if cpi is not None:
        # CPI 温和(1~3%)最有利于股市；过高(>4)压制、过低(<0)通缩
        if cpi > 4:
            v = -0.7
        elif cpi < 0:
            v = -0.5
        elif 1 <= cpi <= 3:
            v = 0.5
        else:
            v = 0.1
        econ_parts.append(("CPI %.1f%%" % cpi, v, 0.7))
    if ppi is not None:
        # PPI 转正=企业盈利改善
        v = max(-1.0, min(1.0, ppi / 3.0))
        econ_parts.append(("PPI %.1f%%" % ppi, v, 0.8))
    if econ_parts:
        tot = sum(w for _, _, w in econ_parts)
        s = sum(x * w for _, x, w in econ_parts) / max(tot, 1e-9)
        out["economy"] = {"score": round((s + 1) / 2, 3),
                          "items": [(n, round(x, 2)) for n, x, _ in econ_parts],
                          "detail": "经济：%s" % "；".join("%s(%+.2f)" % (n, x)
                                                      for n, x, _ in econ_parts)}
    # ---- 政策象限：LPR 方向（下行=宽松）+ M2 增速 ----
    pol_parts = []
    lpr = (ind.get("lpr") or {}).get("value")
    if lpr is not None:
        pol_parts.append(("LPR %.2f%%" % lpr, 0.1, 0.5))
    m2 = (ind.get("m2") or {}).get("value")
    if m2 is not None:
        # M2 增速高于 8% 视为宽松
        v = max(-1.0, min(1.0, (m2 - 8.0) / 3.0))
        pol_parts.append(("M2 %.1f%%" % m2, v, 1.0))
    if pol_parts:
        tot = sum(w for _, _, w in pol_parts)
        s = sum(x * w for _, x, w in pol_parts) / max(tot, 1e-9)
        out["policy"] = {"score": round((s + 1) / 2, 3),
                         "items": [(n, round(x, 2)) for n, x, _ in pol_parts],
                         "detail": "政策/货币：%s" % "；".join("%s(%+.2f)" % (n, x)
                                                         for n, x, _ in pol_parts)}

    # ---- 无法量化的部分：人工核查清单（不编造分数）----
    out["checklist"] = [
        ("政治/地缘", "重要会议与政策定调（中央经济工作会议/两会/政治局会议）、"
                    "中美关系与关税、地缘冲突、台海局势"),
        ("监管态度", "IPO 节奏、减持规则、量化交易监管、退市力度、"
                  "对特定行业（平台/教育/医药/地产）的监管取向"),
        ("社会/人口", "消费信心、就业数据、出生率与老龄化、"
                   "居民资产负债表（是否在提前还贷）"),
        ("产业政策", "产业扶持方向（新质生产力/AI/半导体/新能源）、"
                  "补贴与税收优惠、行业产能治理"),
    ]
    bits = [d.get("detail") for d in (out["policy"], out["economy"])
            if d.get("detail")]
    out["detail"] = ("；".join(bits) if bits
                     else "宏观指标未能取得（akshare 宏观接口不可用）")
    if stale:
        out["detail"] += "　｜ 已排除过期指标：%s" % "、".join(
            "%s(距今%d天>%d天上限)" % (k, a, lim) for k, (a, lim) in stale.items())
    unknown = [k for k, v in out["indicators"].items() if v.get("date_unknown")]
    if unknown:
        out["detail"] += "　｜ 日期缺失（无法校验新鲜度）：%s" % "、".join(unknown)
    return out


def environment_factors(env: dict) -> list:
    out: list = []
    if not env:
        return out
    for key, label, cat in (("economy", "经济基本面", Category.MACRO),
                            ("policy", "政策/流动性", Category.MACRO)):
        d = env.get(key) or {}
        sc = d.get("score")
        if sc is None:
            continue
        d_dir = DIR_BULL if sc >= 0.56 else (DIR_BEAR if sc <= 0.44 else DIR_NEUTRAL)
        out.append(Factor(
            "env_%s" % key, "%s环境 %.0f/100" % (label, sc * 100), cat, d_dir,
            min(abs(sc - 0.5) * 2.6 + 0.2, 0.8), 0.66,
            value="%.0f/100" % (sc * 100),
            detail=(d.get("detail") or "") + "。A股是政策+流动性晴雨表，"
                   "这两项决定中期方向；具体政策事件与地缘风险请对照"
                   "下方人工核查清单判断。",
            theory="宏观周期/政策市", horizon=("d20", "m1", "q1", "h1")))
    return out


# --------------------------------------------------------------------------- #
# 二、主要动能对象与异动意图
# --------------------------------------------------------------------------- #
#: 各宏观指标的新鲜度上限（天）。都是月频/季频数据，留足发布延迟余量。
MACRO_MAX_AGE = {"pmi": 50, "cpi": 70, "ppi": 70, "m2": 70, "lpr": 50}

INTENT_SPEC = {
    "待定": (DIR_NEUTRAL, 0.0, "证据不足：缺少资金流向与板块内宽度数据，"
                              "无法判定意图（不硬判）"),
    "启动": (DIR_BULL, 0.65, "放量启动：资金进场、宽度扩散，属趋势初期"),
    "加速": (DIR_BULL, 0.60, "加速拉升：涨幅与量能同步扩大，动量最强但也最接近尾声"),
    "滞涨": (DIR_BEAR, 0.62, "放量滞涨：涨幅大但资金流出或宽度收缩，派发嫌疑"),
    "护盘": (DIR_NEUTRAL, 0.50, "权重护盘：少数权重拉动指数，多数板块不跟，指数失真"),
    "砸盘": (DIR_BEAR, 0.66, "放量砸盘：权重下跌且量能放大，恐慌或主动打压"),
    "对倒": (DIR_NEUTRAL, 0.45, "对倒制造成交：量能放大但价格几乎不动"),
}


def driver_analysis(boards: list, index_pct: float | None = None) -> dict:
    """识别推动指数涨跌的主要对象，并判定其异动意图。

    ``boards`` 为板块行（含 name/pct/amount/net_inflow/up/down/leader）。
    """
    out = {"items": [], "intent": "", "intent_cn": "", "direction": DIR_NEUTRAL,
           "detail": "", "concentration": None}
    if not boards:
        out["detail"] = "无板块数据，无法识别动能对象"
        return out
    rows = []
    amts = [safe_float(b.get("amount")) for b in boards]
    amts = [a for a in amts if a == a and a > 0]
    total_amt = float(sum(amts)) if amts else 0.0
    for b in boards:
        pct = safe_float(b.get("pct"))
        if pct != pct:
            continue
        amt = safe_float(b.get("amount"))
        share = (amt / total_amt * 100) if (total_amt > 0 and amt == amt) else None
        # 成交额加权涨幅 = 涨跌幅 × 成交额占比 ÷ 100。
        # 注意：这不是对指数点位的真实贡献（真实贡献需市值权重），
        # 只是「按成交额加权的涨跌幅」，用于排序谁对当日拉动最强。
        # 曾因命名「贡献度」被误读为口径不一致——实际 contrib/share = 涨跌幅，
        # 不同板块数值不同是涨跌幅不同所致，公式本身完全自洽。
        contrib = (pct * (share or 0) / 100.0) if share is not None else None
        up = b.get("up") if b.get("up") is not None else b.get("up_count")
        dn = b.get("down") if b.get("down") is not None else b.get("down_count")
        breadth = None
        try:
            if up is not None and dn is not None and (float(up) + float(dn)) > 0:
                breadth = float(up) / (float(up) + float(dn)) * 100
        except Exception:  # noqa: BLE001
            breadth = None
        rows.append({"name": b.get("name"), "pct": pct, "amount": amt,
                     "share": share, "contrib": contrib, "breadth": breadth,
                     "net_inflow": b.get("net_inflow"), "leader": b.get("leader")})
    if not rows:
        out["detail"] = "板块数据无有效涨跌幅"
        return out
    # 按贡献度绝对值排序 —— 对指数影响最大的对象
    rows.sort(key=lambda x: -abs(x.get("contrib") or 0))
    out["items"] = rows[:6]
    top = out["items"][0] if out["items"] else None
    if top and out["items"]:
        shares = [r["share"] for r in out["items"][:3] if r.get("share")]
        out["concentration"] = round(sum(shares), 1) if shares else None
    # ---- 异动意图判定 ----
    ups = sum(1 for r in rows if r["pct"] > 0)
    breadth_ratio = ups / max(len(rows), 1)
    pcts = [r["pct"] for r in rows]
    avg = float(np.mean(pcts)) if pcts else 0.0
    intent = "待定"
    t = top
    # ---- 证据充分性检查（先数证据，再下结论）----
    # 判意图需要三类证据：涨跌幅（必有）、资金净流入、板块内宽度。
    # 新浪板块既无 net_inflow 也常无 up_count/down_count，只有涨跌幅——
    # 这种"单证据"情况下任何意图判定都是猜的，必须如实标注待定。
    n_evidence = 0
    if t:
        if t.get("pct") is not None:
            n_evidence += 1
        if safe_float(t.get("net_inflow")) == safe_float(t.get("net_inflow")) \
                and t.get("net_inflow") is not None:
            n_evidence += 1
        if t.get("breadth") is not None:
            n_evidence += 1
    out["n_evidence"] = n_evidence
    if t and n_evidence < 2:
        d0, s0, desc0 = INTENT_SPEC["待定"]
        out.update({"intent": intent, "intent_cn": intent, "direction": d0,
                    "strength": 0.0, "desc": desc0})
        out["detail"] = (
            "主要动能对象：%s（%+.2f%%，成交占比 %s，贡献 %s）。"
            "**数据不足，意图待定** —— 仅有涨跌幅一项证据，"
            "缺少资金净流入与板块内宽度，任何意图判断都是猜测，故不硬判。"
            % (t["name"], t["pct"],
               ("%.1f%%" % t["share"]) if t.get("share") is not None else "—",
               ("%+.3f" % t["contrib"]) if t.get("contrib") is not None else "—"))
        if out.get("concentration"):
            out["detail"] += "（前三动能对象合计成交占比 %.1f%%）" % out["concentration"]
        return out
    if t:
        big = abs(t["pct"]) >= 2.0
        money_in = (safe_float(t.get("net_inflow")) > 0) if t.get("net_inflow") is not None else None
        wide = (t.get("breadth") or 0) >= 60
        if big and t["pct"] > 0 and (money_in is not False) and wide:
            intent = "启动"
        elif big and t["pct"] > 0 and (money_in is False or not wide):
            intent = "滞涨"
        elif big and t["pct"] < 0:
            intent = "砸盘"
        elif t["pct"] > 0.5 and breadth_ratio <= 0.4:
            intent = "护盘"
        elif big and abs(t["pct"]) < 0.3:
            intent = "对倒"
        elif t["pct"] > 0 and (t.get("breadth") or 100) >= 55:
            intent = "加速" if abs(t["pct"]) >= 3.0 else "启动"
        else:
            intent = "滞涨" if t["pct"] > 0 else "砸盘"
    d, s, desc = INTENT_SPEC[intent]
    out.update({"intent": intent, "intent_cn": intent,
                "direction": d, "strength": s, "desc": desc})
    out["detail"] = (
        "主要动能对象：%s（%+.2f%%，成交占比 %s%%，贡献 %s）→ 异动意图判定为「%s」。%s"
        % (t["name"] if t else "—", t["pct"] if t else 0,
           ("%.1f" % t["share"]) if (t and t.get("share") is not None) else "—",
           ("%+.3f" % t["contrib"]) if (t and t.get("contrib") is not None) else "—",
           intent, desc))
    if out.get("concentration"):
        out["detail"] += "（前三动能对象合计成交占比 %.1f%%）" % out["concentration"]
    return out


def driver_factors(drv: dict) -> list:
    out: list = []
    if not drv or not drv.get("intent"):
        return out
    t = (drv.get("items") or [{}])[0]
    out.append(Factor(
        "market_driver", "涨跌主推手：%s（%s）" % (t.get("name") or "—",
                                              drv["intent"]),
        Category.SENTIMENT, drv.get("direction", DIR_NEUTRAL),
        drv.get("strength", 0.4), 0.6,
        value="%+.2f%%｜成交占比 %s%%" % (
            t.get("pct") or 0,
            ("%.1f" % t["share"]) if t.get("share") is not None else "—"),
        detail=drv.get("detail") or "",
        theory="动能分析/主力异动意图", horizon=("d1", "d5", "d10")))
    return out


if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")

    print("=== 1. 环境四象限 ===")
    env = macro_environment()
    print(" ", env["detail"][:160])
    print("  指标:", {k: v.get("value") for k, v in env["indicators"].items()})
    print("  经济分:", (env["economy"] or {}).get("score"))
    print("  政策分:", (env["policy"] or {}).get("score"))
    print("  核查清单 %d 项" % len(env["checklist"]))
    fs = environment_factors(env)
    print("  环境因子 %d 条: %s" % (len(fs), [f.name for f in fs]))

    print("\n=== 2. 动能对象与意图 ===")
    boards = [
        {"name": "半导体", "pct": 3.6, "amount": 9.0e10, "net_inflow": 55.0,
         "up": 120, "down": 8, "leader": "某龙头"},
        {"name": "银行", "pct": 0.2, "amount": 4.0e10, "net_inflow": -8.0,
         "up": 20, "down": 22, "leader": "—"},
        {"name": "白酒", "pct": -1.8, "amount": 2.5e10, "net_inflow": -30.0,
         "up": 3, "down": 15, "leader": "—"},
    ]
    drv = driver_analysis(boards)
    print(" ", drv["detail"])
    df = driver_factors(drv)
    print("  动能因子 %d 条: %s" % (len(df), [f.name for f in df]))

    print("\n=== 3. 边界 ===")
    assert driver_analysis([])["detail"]
    assert driver_analysis([{"name": "x"}])["detail"]
    assert driver_factors({}) == []
    assert environment_factors({}) == []
    assert macro_environment()["checklist"]
    print("  空输入安全")

    for f in fs + df:
        assert 0 <= f.strength <= 1 and 0 <= f.reliability <= 1
        assert f.detail and f.theory and "None" not in str(f.value)
    print("  因子字段合法")
    print("\nmarket_context self-test OK")
