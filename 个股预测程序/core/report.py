# -*- coding: utf-8 -*-
"""
报告生成 —— 把预测结果变成可读的中文结论（控制台文本 / Markdown / 独立 HTML）。

报告结构
--------
1. **结论速览**：现价、综合评分、操作倾向、8 周期概率表（用户只看这一块也能决策）
2. **核心逻辑**：最重要的看涨/看跌驱动因子（带理论依据与原始观测值）
3. **八维分析**：个股特性/技术面/基本面/资金面/情绪/宏观/消息/周期，逐项展开
4. **理论视角**：道氏、波浪、斐波那契、江恩、量价时空、混沌…
5. **历史验证**：回测 IC、方向胜率、可信度评级（**诚实标注这套方法的局限**）
6. **风险提示**：反向因子、数据缺失项、免责声明
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import os

import pandas as pd

from . import config
from .contracts import Category
from .utils import fmt_num, fmt_pct, get_logger

log = get_logger()

_CSS = """
:root { --up:#c62828; --down:#2e7d32; --bg:#fafafa; --line:#e0e0e0; --ink:#212121; }
* { box-sizing: border-box; }
body { font-family: "Microsoft YaHei","PingFang SC","Noto Sans SC",sans-serif;
       margin: 0; padding: 18px; background: var(--bg); color: var(--ink); line-height: 1.65; }
.wrap { max-width: 1080px; margin: 0 auto; background:#fff; padding: 26px 30px;
        border-radius: 10px; box-shadow: 0 2px 14px rgba(0,0,0,.07); }
h1 { font-size: 23px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 26px 0 10px; padding-left: 10px;
     border-left: 4px solid #1565c0; }
h3 { font-size: 14px; margin: 16px 0 6px; color:#37474f; }
.sub { color:#607d8b; font-size: 13px; margin-bottom: 14px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0 14px; }
th, td { border: 1px solid var(--line); padding: 6px 9px; text-align: center; }
th { background:#f0f4f8; font-weight: 600; }
td.l, th.l { text-align: left; }
tr:nth-child(even) td { background:#fcfcfc; }
.up { color: var(--up); font-weight: 600; }
.down { color: var(--down); font-weight: 600; }
.badge { display:inline-block; padding: 2px 9px; border-radius: 11px; font-size: 12px;
         background:#eceff1; margin-right: 5px; }
.badge.bull { background:#ffebee; color: var(--up); }
.badge.bear { background:#e8f5e9; color: var(--down); }
.card { background:#f7f9fb; border:1px solid var(--line); border-radius:7px;
        padding: 12px 15px; margin: 10px 0; }
.big { font-size: 27px; font-weight: 700; }
.warn { background:#fff8e1; border-left:4px solid #ffa000; padding: 11px 14px;
        border-radius:5px; font-size: 13px; margin: 12px 0; }
.risk { background:#fdecea; border-left:4px solid #c62828; padding: 11px 14px;
        border-radius:5px; font-size: 13px; margin: 12px 0; }
.grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(190px,1fr)); gap:10px; }
.kv { background:#fff; border:1px solid var(--line); border-radius:6px; padding:9px 12px; }
.kv b { display:block; color:#607d8b; font-weight:500; font-size:12px; }
footer { margin-top: 26px; padding-top: 12px; border-top:1px solid var(--line);
         color:#78909c; font-size: 12px; }
"""


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _pct_cls(v: float) -> str:
    if v is None or v != v:
        return ""
    return "up" if v > 0 else ("down" if v < 0 else "")


def _df_to_html(df: pd.DataFrame, left_cols=(), index: bool = False) -> str:
    if df is None or (hasattr(df, "empty") and df.empty):
        return "<p class='sub'>（无数据）</p>"
    d = df.copy()
    classes = []
    for c in d.columns:
        classes.append("l" if c in left_cols else "")
    head = "<tr>" + "".join(
        "<th class='%s'>%s</th>" % (classes[i], _html.escape(str(c)))
        for i, c in enumerate(d.columns)) + "</tr>"
    body = []
    for _, row in d.iterrows():
        cells = []
        for i, c in enumerate(d.columns):
            v = row[c]
            txt = "" if v is None or (isinstance(v, float) and v != v) else str(v)
            cls = classes[i]
            # 自动着色：带 % 的数值列按正负上色
            if isinstance(v, str) and (v.startswith("+") or v.startswith("-")):
                try:
                    num = float(v.replace("%", "").replace(",", ""))
                    cls = (cls + " " + _pct_cls(num)).strip()
                except ValueError:
                    pass
            elif isinstance(v, (int, float)) and c in ("期望涨跌幅", "多空差", "IC", "RankIC"):
                cls = (cls + " " + _pct_cls(float(v))).strip()
            cells.append("<td class='%s'>%s</td>" % (cls, _html.escape(txt)))
        body.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>%s%s</table>" % (head, "".join(body))


def _factor_table(factors, limit: int = 40) -> str:
    if factors is None or not len(factors):
        return "<p class='sub'>（无因子）</p>"
    try:
        df = factors.to_frame()
    except Exception:  # noqa: BLE001
        return "<p class='sub'>（因子表生成失败）</p>"
    if df.empty:
        return "<p class='sub'>（无因子）</p>"
    df = df.copy()
    df["_k"] = df["强度"] * df["可信度"]
    df = df.sort_values("_k", ascending=False).drop(columns=["_k", "id"]).head(limit)
    return _df_to_html(df, left_cols=("因子", "解读", "理论依据", "观测值"))


def _drivers_html(rows) -> str:
    if not rows:
        return "<p class='sub'>（无）</p>"
    items = []
    for r in rows[:10]:
        d = r.get("方向", "看涨")
        cls = "bull" if d == "看涨" else "bear"
        items.append(
            "<div class='card'><span class='badge %s'>%s</span>"
            "<b>%s</b> <span class='sub'>[%s] 强度%.2f/可信度%.2f 贡献%+.1f%%</span>"
            "<div>%s</div><div class='sub'>观测值：%s</div></div>" % (
                cls, _html.escape(str(d)), _html.escape(str(r.get("因子", ""))),
                _html.escape(str(r.get("类别", ""))), float(r.get("强度", 0) or 0),
                float(r.get("可信度", 0) or 0), float(r.get("贡献", 0) or 0),
                _html.escape(str(r.get("解读", ""))),
                _html.escape(str(r.get("观测值", "—")))))
    return "".join(items)


# --------------------------------------------------------------------------- #
# HTML 报告
# --------------------------------------------------------------------------- #
def build_html(result, *, bt_table: pd.DataFrame | None = None,
               grade: dict | None = None, calibration: dict | None = None,
               include_factors: bool = True) -> str:
    """生成完整的独立 HTML 报告字符串。"""
    b = getattr(result, "bundle", None)
    ens = getattr(result, "ensemble", None) or {}
    name = result.name or result.symbol
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts: list[str] = []
    A = parts.append

    A("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>")
    A("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    A("<title>%s(%s) 涨跌概率预测报告</title>" % (_html.escape(name), result.symbol))
    A("<style>%s</style></head><body><div class='wrap'>" % _CSS)

    # ---- 标题 ---------------------------------------------------------- #
    A("<h1>%s（%s）涨跌概率预测报告</h1>" % (_html.escape(name), result.symbol))
    A("<div class='sub'>数据截止：%s ｜ 现价：%s ｜ 报告生成：%s</div>" % (
        result.as_of or "—",
        ("%.2f" % result.price) if result.price == result.price else "—", now))

    # ---- 结论速览 ------------------------------------------------------ #
    stance = ens.get("stance", "—")
    score = ens.get("score_pct", 0.0)
    cons = ens.get("consistency", {}) or {}
    A("<div class='card'><div class='grid'>")
    A("<div class='kv'><b>综合评分</b><span class='big %s'>%+.1f</span> <span class='sub'>/100</span></div>"
      % (_pct_cls(score), score))
    A("<div class='kv'><b>操作倾向</b><span class='big'>%s</span></div>" % _html.escape(str(stance)))
    f20 = result.by_key("d20")
    if f20:
        A("<div class='kv'><b>未来20天上涨概率</b><span class='big %s'>%.1f%%</span></div>"
          % ("up" if f20.p_up >= 0.5 else "down", f20.p_up * 100))
        A("<div class='kv'><b>未来20天期望涨跌</b><span class='big %s'>%+.2f%%</span>"
          "<div class='sub'>区间 %+.1f%% ~ %+.1f%%</div></div>"
          % (_pct_cls(f20.exp_return), f20.exp_return, f20.ci_low, f20.ci_high))
    A("</div>")
    if cons.get("detail"):
        A("<div class='sub' style='margin-top:9px'>%s</div>" % _html.escape(str(cons["detail"])))
    A("</div>")

    # ---- 8 周期概率表 -------------------------------------------------- #
    A("<h2>一、八大周期概率预测</h2>")
    if result.forecasts:
        fdf = result.to_frame()
        # 补充幅度与模型信息
        fdf["趋势强度"] = ["%+.0f" % f.strength_score for f in result.forecasts]
        fdf["AI参与度"] = [("%.0f%%" % (f.model_mix.get("AI模型", 0) * 100))
                          for f in result.forecasts]
        A(_df_to_html(fdf, left_cols=("周期", "结论")))
        A("<h3>逐周期解读</h3>")
        for f in result.forecasts:
            A("<div class='card'><b>%s —— %s</b>"
              "<div>上涨概率 <span class='%s'>%.1f%%</span>，下跌概率 %.1f%%；"
              "期望涨跌幅 <span class='%s'>%+.2f%%</span>"
              "（约 72%% 概率落在 %+.1f%% ~ %+.1f%%）；"
              "大涨(>%.1f%%)概率 %.1f%%，大跌概率 %.1f%%。</div>"
              "<div class='sub'>置信度 %.0f%%｜趋势强度 %+.1f｜预测波动率 %.2f%%%s</div></div>" % (
                  _html.escape(f.label), _html.escape(f.trend),
                  "up" if f.p_up >= 0.5 else "down", f.p_up * 100, f.p_down * 100,
                  _pct_cls(f.exp_return), f.exp_return, f.ci_low, f.ci_high,
                  f.big_move_threshold, f.p_big_up * 100, f.p_big_down * 100,
                  f.confidence * 100, f.strength_score, f.sigma,
                  ("｜" + _html.escape(f.note)) if f.note else ""))
    else:
        A("<p class='sub'>未生成预测。</p>")

    # ---- 核心逻辑 ------------------------------------------------------ #
    A("<h2>二、核心逻辑（主要驱动与反向风险）</h2>")
    if ens.get("top_drivers"):
        A("<h3>主要看涨/主导驱动</h3>")
        A("<p>%s</p>" % "、".join("<span class='badge bull'>%s</span>" % _html.escape(d["因子"])
                                 for d in ens["top_drivers"]))
    if ens.get("top_risks"):
        A("<h3>主要反向风险</h3>")
        A("<p>%s</p>" % "、".join("<span class='badge bear'>%s</span>" % _html.escape(d["因子"])
                                 for d in ens["top_risks"]))
    if ens.get("key_levels"):
        A("<h3>关键价位参考（基于未来20天预测）</h3>")
        A("<div class='grid'>")
        for k, v in ens["key_levels"].items():
            A("<div class='kv'><b>%s</b>%.2f</div>" % (_html.escape(k), float(v)))
        A("</div>")
    if f20 and (f20.drivers or f20.counter):
        A("<h3>未来20天因子明细</h3>")
        A("<h4 style='margin:9px 0 3px'>正向贡献</h4>")
        A(_drivers_html(f20.drivers))
        if f20.counter:
            A("<h4 style='margin:9px 0 3px'>反向贡献</h4>")
            A(_drivers_html(f20.counter))

    # ---- 八维分析 ------------------------------------------------------ #
    A("<h2>三、八维因素分析</h2>")
    A(_dimensions_html(b))

    # ---- 理论视角 ------------------------------------------------------ #
    A("<h2>四、金融理论视角</h2>")
    A(_theory_html(b))

    # ---- 历史验证 ------------------------------------------------------ #
    A("<h2>五、历史验证（这套方法过去有多准）</h2>")
    if grade:
        A("<div class='card'><b>可信度评级：%s</b>（评分 %.2f）"
          "<div>%s</div><div class='sub'>平均 IC=%s，平均方向胜率=%s，正相关周期占比=%s</div></div>" % (
              _html.escape(str(grade.get("grade"))), float(grade.get("score", 0)),
              _html.escape(str(grade.get("detail", ""))),
              grade.get("mean_ic"), grade.get("mean_hit"), grade.get("positive_ratio")))
    if bt_table is not None and not (hasattr(bt_table, "empty") and bt_table.empty):
        A(_df_to_html(bt_table, left_cols=("周期", "说明")))
    else:
        A("<p class='sub'>未做回测（历史样本不足）。</p>")
    if calibration:
        A("<h3>置信度校准（依据历史表现折扣）</h3>")
        rows = [{"周期": result.by_key(k).label if result.by_key(k) else k,
                 "校准系数": v.get("factor"), "历史RankIC": v.get("rank_ic"),
                 "历史胜率": (("%.1f%%" % (v["hit_rate"] * 100))
                              if v.get("hit_rate") is not None else "—"),
                 "样本数": v.get("n"), "原因": v.get("reason")}
                for k, v in calibration.items()]
        A(_df_to_html(pd.DataFrame(rows), left_cols=("周期", "原因")))

    # ---- 风险提示 ------------------------------------------------------ #
    A("<h2>六、风险提示与免责声明</h2>")
    notes = list(getattr(b, "notes", []) or []) + list(result.meta.get("notes") or [])
    errors = list(result.meta.get("errors") or []) or list(getattr(b, "errors", []) or [])
    if notes:
        A("<div class='warn'><b>数据说明：</b><ul>%s</ul></div>"
          % "".join("<li>%s</li>" % _html.escape(str(n)) for n in dict.fromkeys(notes)))
    if errors:
        A("<div class='warn'><b>以下模块未能完成分析（不影响其它维度）：</b><ul>%s</ul></div>"
          % "".join("<li>%s</li>" % _html.escape(str(e)) for e in dict.fromkeys(errors)))
    A("<div class='risk'>本报告由量化模型自动生成，所有概率与幅度均为**统计推断**，"
      "不构成任何投资建议。A 股市场受政策、资金与情绪影响极大，历史规律随时可能失效；"
      "模型对突发事件（黑天鹅）无能为力。请独立判断并自行承担投资风险。</div>")

    A("<footer>生成时间 %s ｜ 模型版本 %s ｜ 因子总数 %d ｜ 数据来源 %s</footer>" % (
        now, _version(), len(getattr(b, "factors", []) or []),
        _html.escape(str(result.meta.get("sources", {})))))
    A("</div></body></html>")
    return "".join(parts)


def _version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return "1.0.0"


def _dimensions_html(b) -> str:
    """八维因素的分析摘要（从 bundle 的富数据字段里取）。"""
    if b is None:
        return "<p class='sub'>（无明细）</p>"
    out: list[str] = []

    # 1) 个股特性与阶段
    st = b.stage or {}
    snap = b.snapshot or {}
    out.append("<h3>1. 个股特性与所处阶段</h3>")
    if snap or st:
        kv = []
        for label, key in (("名称", "name"), ("现价", "price"), ("涨跌幅", "pct_chg"),
                           ("换手率", "turnover"), ("量比", "volume_ratio"),
                           ("总市值", "market_cap"), ("流通市值", "float_market_cap"),
                           ("PE(TTM)", "pe_ttm"), ("PB", "pb"), ("振幅", "amplitude")):
            v = snap.get(key)
            if v is None:
                continue
            if key in ("market_cap", "float_market_cap"):
                v = fmt_num(float(v))
            kv.append("<div class='kv'><b>%s</b>%s</div>" % (label, _html.escape(str(v))))
        for label, key in (("所处阶段", "stage"), ("阶段说明", "stage_detail"),
                           ("定价权", "pricing_power"), ("量化活跃度", "quant_level"),
                           ("趋势特征", "trend_type"), ("控盘程度", "control_level"),
                           ("筹码集中度", "chip_concentration")):
            v = st.get(key) or ((st.get("control_degree") or {}).get(key))
            if v is None:
                continue
            kv.append("<div class='kv'><b>%s</b>%s</div>" % (label, _html.escape(str(v))))
        out.append("<div class='grid'>%s</div>" % "".join(kv))
        detail = st.get("detail") or st.get("stage_detail")
        if detail:
            out.append("<p>%s</p>" % _html.escape(str(detail)))
    else:
        out.append("<p class='sub'>（未取得个股特性数据）</p>")

    # 2) 技术面
    out.append("<h3>2. 技术面</h3>")
    ind = b.indicators or {}
    tech_lines = []
    for key, label in (("trend", "趋势"), ("ma_state", "均线系统"), ("vol_state", "量能"),
                       ("summary", "技术面小结"), ("divergence", "背离")):
        v = ind.get(key)
        if v:
            tech_lines.append("<b>%s</b>：%s" % (label, _html.escape(str(v))))
    if tech_lines:
        out.append("<div class='card'>%s</div>" % "<br>".join(tech_lines))
    sr = ind.get("support_resistance") or ind.get("sr")
    if isinstance(sr, dict) and (sr.get("supports") or sr.get("resistances")):
        sup = "、".join("%.2f" % (x.get("price") if isinstance(x, dict) else x)
                        for x in (sr.get("supports") or [])[:4])
        res = "、".join("%.2f" % (x.get("price") if isinstance(x, dict) else x)
                        for x in (sr.get("resistances") or [])[:4])
        out.append("<p>支撑位：%s ｜ 压力位：%s</p>" % (sup or "—", res or "—"))

    # 3) 基本面
    out.append("<h3>3. 基本面</h3>")
    val = b.valuation or {}
    if val:
        kv = []
        for label, key in (("PE(TTM)", "pe_ttm"), ("PB", "pb"), ("PE历史分位", "pe_percentile"),
                           ("PB历史分位", "pb_percentile"), ("总市值", "market_cap"),
                           ("股息率", "dividend_yield")):
            v = val.get(key)
            if v is None:
                continue
            if key.endswith("percentile") and isinstance(v, (int, float)):
                v = "%.1f%%" % (float(v) * 100)
            elif isinstance(v, (int, float)) and abs(v) > 1e6:
                v = fmt_num(float(v))
            kv.append("<div class='kv'><b>%s</b>%s</div>" % (label, _html.escape(str(v))))
        out.append("<div class='grid'>%s</div>" % "".join(kv))
    fin = b.financials
    if isinstance(fin, pd.DataFrame) and not fin.empty:
        cols = [c for c in ("报告期", "营业收入", "营收同比", "净利润", "净利同比", "ROE",
                            "毛利率", "净利率", "资产负债率", "经营现金流") if c in fin.columns]
        if cols:
            out.append(fin[cols].tail(6).to_string(index=False) and
                       _df_to_html(fin[cols].tail(6), left_cols=("报告期",)))
    else:
        out.append("<p class='sub'>（未取得财务数据）</p>")

    # 4) 资金面
    out.append("<h3>4. 资金面与主力动向</h3>")
    cap = b.capital or {}
    lines = []
    flow = cap.get("flow")
    if isinstance(flow, pd.DataFrame) and not flow.empty:
        try:
            last = flow.tail(5)
            net = float(last["main_net"].sum()) if "main_net" in last.columns else float("nan")
            lines.append("近 5 日主力资金净流入：<b>%s</b>" % fmt_num(net, unit="元"))
            if "main_net_20d" in flow.columns:
                lines.append("近 20 日累计：<b>%s</b>"
                             % fmt_num(float(flow["main_net_20d"].iloc[-1]), unit="元"))
        except Exception:  # noqa: BLE001
            pass
    prof = cap.get("main_force_profile") or cap.get("profile")
    if isinstance(prof, dict) and prof.get("behavior"):
        lines.append("主力行为判定：<b>%s</b>（置信度 %.0f%%）%s"
                     % (_html.escape(str(prof["behavior"])),
                        float(prof.get("confidence", 0)) * 100,
                        _html.escape(str(prof.get("detail", "")))))
    holders = cap.get("holders")
    if isinstance(holders, pd.DataFrame) and not holders.empty:
        try:
            lines.append("最新股东户数：%s" % _html.escape(str(holders.iloc[-1].to_dict())))
        except Exception:  # noqa: BLE001
            pass
    lhb = cap.get("lhb")
    if isinstance(lhb, pd.DataFrame) and not lhb.empty:
        lines.append("近期龙虎榜记录 %d 条" % len(lhb))
    out.append("<div class='card'>%s</div>" % ("<br>".join(lines) if lines
                                               else "<span class='sub'>（未取得资金流数据）</span>"))

    # 5) 情绪
    out.append("<h3>5. 市场情绪与博弈心理</h3>")
    sent = b.sentiment or {}
    slines = []
    fg = sent.get("fear_greed") or {}
    if isinstance(fg, dict) and fg.get("value") is not None:
        slines.append("恐慌/贪婪指数：<b>%s</b>（%s）"
                      % (fg.get("value"), _html.escape(str(fg.get("level", "")))))
    mkt = sent.get("market") or {}
    if isinstance(mkt, dict) and mkt:
        for k, lab in (("limit_up_count", "涨停家数"), ("limit_down_count", "跌停家数"),
                       ("up_count", "上涨家数"), ("down_count", "下跌家数"),
                       ("max_continuous", "最高连板"), ("broken_rate", "炸板率")):
            v = mkt.get(k)
            if v is None:
                continue
            if isinstance(v, float) and v != v:      # NaN 跳过，不显示 "nan"
                continue
            if k == "broken_rate":
                slines.append("%s：<b>%.1f%%</b>" % (lab, float(v) * (100 if float(v) <= 1 else 1)))
            else:
                slines.append("%s：<b>%s</b>" % (lab, v))
    sec = sent.get("sector") or {}
    if isinstance(sec, dict) and sec.get("boards"):
        names = [b2.get("name") for b2 in sec["boards"][:8] if isinstance(b2, dict)]
        names = [n for n in names if n]
        if names:
            slines.append("所属板块：" + _html.escape("、".join(str(n) for n in names)))
    retail = sent.get("retail") or {}
    if isinstance(retail, dict) and retail:
        label_map = {"guba_rank": "股吧人气排名", "guba_rank_change": "人气排名变化",
                     "search_hits": "搜索热度", "search_hits_change_pct": "搜索热度变化%",
                     "guba_bull_ratio": "股吧多空比"}
        bits = []
        for k, v in retail.items():
            if isinstance(v, (dict, list)) or v is None:
                continue
            if isinstance(v, float) and v != v:
                continue
            bits.append("%s=%s" % (label_map.get(k, k), v))
        if bits:
            slines.append("散户行为（反向参考）：" + _html.escape("；".join(bits)))
    out.append("<div class='card'>%s</div>" % ("<br>".join(slines) if slines
                                               else "<span class='sub'>（未取得情绪数据）</span>"))

    # 6) 宏观
    out.append("<h3>6. 宏观与行业环境</h3>")
    mac = b.macro or {}
    mlines = []
    reg = mac.get("market_regime") or {}
    if isinstance(reg, dict) and reg:
        mlines.append("大盘状态：<b>%s</b>（%s）｜年化波动 %.1f%%｜距高点回撤 %.1f%%"
                      % (_html.escape(str(reg.get("regime", ""))),
                         _html.escape(str(reg.get("ma_state", ""))),
                         float(reg.get("annual_vol_pct") or 0),
                         float(reg.get("drawdown_from_high_pct") or 0)))
    mm = mac.get("macro") or {}
    if isinstance(mm, dict) and mm:
        keys = [("gdp_yoy", "GDP同比"), ("cpi_yoy", "CPI"), ("pmi", "PMI"),
                ("lpr_1y", "LPR(1年)"), ("m2_yoy", "M2同比"), ("shibor_3m", "3月Shibor")]
        bits = ["%s=%s" % (lab, mm.get(k)) for k, lab in keys if mm.get(k) is not None]
        if bits:
            mlines.append("宏观指标：" + _html.escape("；".join(bits)))
    pol = mac.get("policy") or {}
    if isinstance(pol, dict) and pol.get("detail"):
        mlines.append("产业政策：" + _html.escape(str(pol["detail"])))
    # 上下游商品：只写"按行业筛选后"的相关品种，避免出现与个股无关的商品
    rel = mac.get("commodity_relevant") or {}
    rel_items = rel.get("items") or {}
    if isinstance(rel, dict) and rel.get("detail"):
        mlines.append("上下游商品：" + _html.escape(str(rel["detail"]))
                      + ("（" + "、".join(
                          "%s %+.2f%%" % (k, float((v or {}).get("pct_chg") or 0))
                          for k, v in list(rel_items.items())[:4]) + "）"
                         if rel_items else ""))
    elif isinstance(mac.get("commodity"), dict) and mac["commodity"]:
        mlines.append("上下游商品：" + _html.escape(
            str(mac["commodity"].get("detail") or "已取到商品价格")))
    out.append("<div class='card'>%s</div>" % ("<br>".join(mlines) if mlines
                                               else "<span class='sub'>（未取得宏观数据）</span>"))

    # 7) 消息面
    out.append("<h3>7. 消息面与突发事件</h3>")
    news = b.news or []
    if news:
        rows = []
        for n in news[:15]:
            if not isinstance(n, dict):
                continue
            d = n.get("direction", 0)
            rows.append({"日期": str(n.get("date", ""))[:10],
                         "标题": str(n.get("title", ""))[:60],
                         "标签": n.get("tag", ""),
                         "倾向": "利好" if d and d > 0 else ("利空" if d and d < 0 else "中性")})
        if rows:
            out.append(_df_to_html(pd.DataFrame(rows), left_cols=("标题", "日期", "标签")))
    else:
        out.append("<p class='sub'>（未取得近期新闻/公告）</p>")

    # 8) 周期
    out.append("<h3>8. 时间周期与季节性</h3>")
    sea = b.seasonality or {}
    if sea:
        rows = []
        for k, v in sea.items():
            if k.startswith("_") or isinstance(v, (dict, list)):
                continue
            rows.append({"项目": k, "值": v})
        if rows:
            out.append(_df_to_html(pd.DataFrame(rows), left_cols=("项目",)))
        cal = sea.get("calendar") or sea.get("calendar_effects")
        if isinstance(cal, dict) and cal.get("effects"):
            items = cal["effects"]
            out.append("<p>%s</p>" % "；".join(
                "%s（%s）" % (_html.escape(str(e.get("name"))),
                             _html.escape(str(e.get("detail", "")))) for e in items[:6]))
    else:
        out.append("<p class='sub'>（未取得季节性数据）</p>")
    return "".join(out)


def _theory_html(b) -> str:
    if b is None or not (b.theory or {}):
        return "<p class='sub'>（未取得理论模型结果）</p>"
    th = b.theory or {}
    out = []
    mapping = [
        ("dow", "道氏理论", ["trend", "phase", "volume_confirm", "detail"]),
        ("elliott", "艾略特波浪理论", ["wave", "confidence", "detail"]),
        ("fib", "斐波那契理论", ["current_zone", "nearest_support", "nearest_resistance", "detail"]),
        ("gann", "江恩理论", ["score", "confidence", "detail"]),
        ("vpt", "量价时空理论", ["volume_stage", "price_stage", "match", "detail"]),
        ("fan", "扇形原理", ["current_line", "position", "detail"]),
        ("pct", "百分比回撤原理", ["nearest_level", "position_pct", "detail"]),
        ("chaos", "混沌理论", ["hurst", "fractal_dimension", "market_state", "detail"]),
        ("resonance", "多周期共振", ["resonance", "resonance_direction", "strength", "detail"]),
    ]
    for key, label, fields in mapping:
        d = th.get(key)
        if not isinstance(d, dict) or not d:
            continue
        bits = []
        for f in fields:
            v = d.get(f)
            if v is None:
                continue
            if isinstance(v, float):
                v = "%.3f" % v
            bits.append("<b>%s</b>=%s" % (f, _html.escape(str(v))))
        if bits:
            out.append("<div class='card'><b>%s</b><div>%s</div></div>"
                       % (label, " ｜ ".join(bits)))
    cons = th.get("consensus") or {}
    if isinstance(cons, dict) and cons.get("detail"):
        out.append("<div class='card'><b>理论综合</b>：%s（看涨 %s / 看跌 %s）</div>"
                   % (_html.escape(str(cons["detail"])), cons.get("bull_count"),
                      cons.get("bear_count")))
    return "".join(out) if out else "<p class='sub'>（理论模型无结论）</p>"


# --------------------------------------------------------------------------- #
# 文本报告
# --------------------------------------------------------------------------- #
def build_text(result, *, bt_table: pd.DataFrame | None = None,
               grade: dict | None = None, width: int = 100) -> str:
    """生成适合控制台/Markdown 的文本报告。"""
    b = getattr(result, "bundle", None)
    ens = getattr(result, "ensemble", None) or {}
    L: list[str] = []
    sep = "=" * width

    L.append(sep)
    L.append("%s（%s） 涨跌概率预测报告" % (result.name or "", result.symbol))
    L.append("数据截止 %s ｜ 现价 %s ｜ 生成 %s" % (
        result.as_of or "—",
        ("%.2f" % result.price) if result.price == result.price else "—",
        _dt.datetime.now().strftime("%Y-%m-%d %H:%M")))
    L.append(sep)

    L.append("")
    L.append("【结论速览】综合评分 %+.1f/100 ｜ 操作倾向：%s" % (
        ens.get("score_pct", 0.0), ens.get("stance", "—")))
    cons = ens.get("consistency") or {}
    if cons.get("detail"):
        L.append("            %s" % cons["detail"])
    L.append("")
    L.append("【八大周期预测】")
    if result.forecasts:
        try:
            L.append(result.to_frame().to_string(index=False))
        except Exception:  # noqa: BLE001
            for f in result.forecasts:
                L.append("  " + f.summary())
    L.append("")
    L.append("【逐周期解读】")
    for f in result.forecasts:
        L.append("  · %s：%s" % (f.label, f.trend))
        L.append("      上涨 %.1f%% / 下跌 %.1f%%｜期望 %+.2f%%（区间 %+.1f%%~%+.1f%%）"
                 "｜大涨 %.1f%% / 大跌 %.1f%%｜置信度 %.0f%%" % (
                     f.p_up * 100, f.p_down * 100, f.exp_return, f.ci_low, f.ci_high,
                     f.p_big_up * 100, f.p_big_down * 100, f.confidence * 100))
        if f.note:
            L.append("      注：%s" % f.note)

    if ens.get("top_drivers") or ens.get("top_risks"):
        L.append("")
        L.append("【核心驱动】")
        for d in ens.get("top_drivers", []):
            L.append("  ▲ %s（权重分 %.2f）" % (d["因子"], d["权重分"]))
        for d in ens.get("top_risks", []):
            L.append("  ▼ %s（权重分 %.2f）" % (d["因子"], d["权重分"]))

    if ens.get("key_levels"):
        L.append("")
        L.append("【关键价位（未来20天）】")
        for k, v in ens["key_levels"].items():
            L.append("  %-14s %.2f" % (k, float(v)))

    f20 = result.by_key("d20")
    if f20 and (f20.drivers or f20.counter):
        L.append("")
        L.append("【未来20天因子明细】")
        for r in f20.drivers:
            L.append("  + %-18s %s（强度%.2f 可信度%.2f 贡献%+.1f%%）" % (
                r["因子"], r["解读"][:46], r["强度"], r["可信度"], r["贡献"]))
        for r in f20.counter:
            L.append("  - %-18s %s（强度%.2f 可信度%.2f 贡献%+.1f%%）" % (
                r["因子"], r["解读"][:46], r["强度"], r["可信度"], r["贡献"]))

    L.append("")
    L.append(sep)
    L.append("【历史验证】")
    if grade:
        L.append("  可信度评级：%s（评分 %.2f）" % (grade.get("grade"), grade.get("score", 0)))
        L.append("  %s" % grade.get("detail", ""))
    if bt_table is not None and not (hasattr(bt_table, "empty") and bt_table.empty):
        try:
            cols = [c for c in ("周期", "样本数", "IC", "RankIC", "方向胜率", "多空差")
                    if c in bt_table.columns]
            L.append(bt_table[cols].to_string(index=False))
        except Exception:  # noqa: BLE001
            pass
    else:
        L.append("  未做回测（历史样本不足）")

    L.append("")
    L.append(sep)
    L.append("【数据说明】")
    for n in dict.fromkeys(list(getattr(b, "notes", []) or []) +
                           list(result.meta.get("notes") or [])):
        L.append("  · %s" % n)
    errs = list(result.meta.get("errors") or []) or list(getattr(b, "errors", []) or [])
    if errs:
        L.append("【未能完成的模块】")
        for e in dict.fromkeys(errs):
            L.append("  ! %s" % e)
    if b is not None:
        by_cat = (b.stats().get("by_category") or {})
        cat_txt = "、".join("%s%d" % (k, sum(v.values()) if isinstance(v, dict) else v)
                           for k, v in by_cat.items())
        L.append("  因子总数：%d（%s）" % (len(b.factors), cat_txt or "—"))
    L.append("  数据来源：%s" % result.meta.get("sources", {}))
    L.append("")
    L.append("【免责声明】本报告由量化模型自动生成，概率与幅度均为统计推断，不构成投资建议。")
    L.append("            A股受政策/资金/情绪影响极大，历史规律随时可能失效，请自行承担风险。")
    L.append(sep)
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #
def save_report(result, *, out_dir: str | None = None, bt_table=None, grade=None,
                calibration=None, formats=("html", "md", "txt")) -> dict:
    """把报告保存到文件，返回 ``{"html": path, ...}``。"""
    out_dir = out_dir or config.OUTPUT["report_dir"]
    os.makedirs(out_dir, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = "%s_%s" % (result.symbol or "stock", stamp)
    paths: dict[str, str] = {}
    try:
        if "html" in formats:
            p = os.path.join(out_dir, base + ".html")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(build_html(result, bt_table=bt_table, grade=grade,
                                    calibration=calibration))
            paths["html"] = p
        text = build_text(result, bt_table=bt_table, grade=grade)
        if "md" in formats:
            p = os.path.join(out_dir, base + ".md")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("```\n" + text + "\n```\n")
            paths["md"] = p
        if "txt" in formats:
            p = os.path.join(out_dir, base + ".txt")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text)
            paths["txt"] = p
    except Exception as exc:  # noqa: BLE001
        log.warning("报告保存失败: %s", exc)
    return paths


if __name__ == "__main__":  # pragma: no cover
    import tempfile

    from .engine import quick_predict

    res = quick_predict("600519", use_ml=False, demo=True)
    txt = build_text(res)
    print(txt[:1800])
    print("\n...（中略）...\n")
    html = build_html(res)
    print("HTML 长度 %d 字符，包含关键区块：%s" % (
        len(html), all(k in html for k in ("八大周期概率预测", "八维因素分析", "金融理论视角",
                                           "历史验证", "免责声明"))))
    assert "<table>" in html and "上涨概率" in html
    # 落盘测试（写工作区内目录，避免文件沙箱限制）
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_report_selftest")
    paths = save_report(res, out_dir=out, formats=("html", "txt"))
    for k, v in paths.items():
        print("  %s -> %s (%d bytes)" % (k, v, os.path.getsize(v)))
    assert "html" in paths and os.path.getsize(paths["html"]) > 3000
    print("report self-test OK")
