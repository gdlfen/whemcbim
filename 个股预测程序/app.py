# -*- coding: utf-8 -*-
"""
个股涨跌概率预测 —— Streamlit 前端 + 后端入口。

部署说明
--------
* **本地运行**：在本文件所在目录执行 ``streamlit run app.py``
* **Streamlit Community Cloud**：仓库根目录放本文件即可，``main file path`` 填 ``app.py``；
  依赖写在 ``requirements.txt``；``packages.txt`` 用于安装中文字体（可选）。
* 本文件既是"部署入口"也是"后端入口"：真正的计算全部在 ``core/`` 包里，
  与本地版完全共用同一套代码，不存在两份逻辑。

移动端适配
----------
* ``layout="wide"`` + 单列/双列自适应；表格放在可横向滚动容器里；
* 关键结论用大字号指标卡，手机上一屏可见；
* 长文本统一放进 ``st.expander`` 折叠。
"""

from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# 保证 core 包可导入（兼容两种目录结构：
#   1) app.py 与 core/ 同级（本仓库的推荐结构）
#   2) app.py 在 webapp/ 下、core/ 在上一级
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE, os.path.dirname(_HERE), os.getcwd()):
    if _cand and os.path.isdir(os.path.join(_cand, "core")) and _cand not in sys.path:
        sys.path.insert(0, _cand)

st.set_page_config(
    page_title="个股涨跌概率预测",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"about": "基于技术面/基本面/资金面/情绪面/宏观/消息面/周期/理论模型的多周期概率预测。"
                         "仅供研究，不构成投资建议。"},
)

# --------------------------------------------------------------------------- #
# 样式（移动端友好）
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
  .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1180px; }
  div[data-testid="stMetricValue"] { font-size: 1.45rem; }
  .bigprobs { display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 2px; }
  .pcard { flex:1 1 118px; border:1px solid #e3e8ee; border-radius:9px; padding:9px 11px;
           background:#fbfcfe; }
  .pcard .lbl { font-size:12px; color:#5b6b7c; }
  .pcard .val { font-size:19px; font-weight:700; }
  .pcard .sub { font-size:11px; color:#7b8a99; }
  .up { color:#c62828; } .down { color:#2e7d32; } .flat { color:#607d8b; }
  .pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11.5px;
          background:#eceff1; margin:1px 3px 1px 0; }
  .pill.bull { background:#ffebee; color:#c62828; }
  .pill.bear { background:#e8f5e9; color:#2e7d32; }
  .scrollx { overflow-x:auto; -webkit-overflow-scrolling:touch; }
  .scrollx table { font-size:12.5px; }
  .note { background:#fff8e1; border-left:4px solid #ffa000; padding:9px 12px;
          border-radius:5px; font-size:12.5px; margin:8px 0; }
  .risk { background:#fdecea; border-left:4px solid #c62828; padding:9px 12px;
          border-radius:5px; font-size:12.5px; margin:8px 0; }
  @media (max-width: 640px) {
    .block-container { padding-left: .7rem; padding-right: .7rem; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; }
    .pcard { flex:1 1 46%; }
  }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# 核心库导入
# --------------------------------------------------------------------------- #
CORE_ERROR = None
try:
    from core import __version__ as CORE_VERSION
    from core import backtest as bt_mod
    from core import charts as charts_mod
    from core import config as cfg
    from core import report as report_mod
    from core.contracts import HORIZONS, Category, horizon_spec
    from core.engine import Predictor
    from core.utils import cache_clear, get_logger, normalize_symbol
except Exception as _exc:  # noqa: BLE001
    CORE_ERROR = traceback.format_exc()
    CORE_VERSION = "?"

log = get_logger()

DISCLAIMER = ("本页面由量化模型自动生成，所有概率与涨跌幅度均为**统计推断**，"
              "不构成任何投资建议。A 股受政策、资金与情绪影响极大，"
              "历史规律随时可能失效；模型对黑天鹅事件无能为力。请独立判断并自行承担风险。")


# --------------------------------------------------------------------------- #
# 数据/分析缓存
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=1800, max_entries=24)
def cached_analysis(symbol: str, use_ml: bool, demo: bool, days: int) -> dict:
    """执行一次完整分析并返回可序列化的结果字典。

    注意：matplotlib 图对象与 DataFrame 不放进缓存（体积大、可能不可序列化），
    只缓存**结论性数据**；图表每次由轻量函数现场生成。
    """
    t0 = time.time()
    predictor = Predictor(symbol, use_ml=use_ml, demo=demo, days=days, verbose=False)

    stages: list = []

    def _progress(stage: str, pct: float) -> None:
        stages.append((stage, pct))

    result = predictor.run(progress=_progress)
    bundle = result.bundle

    payload = {
        "symbol": result.symbol,
        "name": result.name,
        "as_of": result.as_of,
        "price": float(result.price) if result.price == result.price else None,
        "elapsed": round(time.time() - t0, 1),
        "forecasts": [
            {
                "horizon": f.horizon, "label": f.label, "trend": f.trend,
                "p_up": f.p_up, "p_down": f.p_down,
                "p_big_up": f.p_big_up, "p_big_down": f.p_big_down, "p_flat": f.p_flat,
                "exp_return": f.exp_return, "exp_low": f.exp_return_low,
                "exp_high": f.exp_return_high, "ci_low": f.ci_low, "ci_high": f.ci_high,
                "sigma": f.sigma, "big_move": f.big_move_threshold,
                "confidence": f.confidence, "strength_score": f.strength_score,
                "model_mix": dict(f.model_mix or {}), "note": f.note,
                "drivers": list(f.drivers or []), "counter": list(f.counter or []),
            } for f in result.forecasts
        ],
        "ensemble": result.ensemble or {},
        "meta_sources": result.meta.get("sources", {}),
        "meta_errors": list(result.meta.get("errors") or []),
        "notes": list(dict.fromkeys(list(result.meta.get("notes") or [])
                                    + list(getattr(bundle, "notes", []) or []))),
        "is_demo": bool(result.meta.get("demo")),
        "n_factors": len(getattr(bundle, "factors", []) or []),
        "factor_stats": (bundle.stats() if bundle is not None else {}),
        "volatility": (bundle.indicators.get("volatility") if bundle is not None else {}),
        "snapshot": _clean_dict(getattr(bundle, "snapshot", {}) or {}),
        "stage": _clean_dict(getattr(bundle, "stage", {}) or {}),
        "valuation": _clean_dict(getattr(bundle, "valuation", {}) or {}),
        "capital_brief": _capital_brief(getattr(bundle, "capital", {}) or {}),
        "sentiment": _deep_clean(getattr(bundle, "sentiment", {}) or {}, depth=2),
        "macro": _deep_clean(getattr(bundle, "macro", {}) or {}, depth=2),
        "news": [n for n in (getattr(bundle, "news", []) or []) if isinstance(n, dict)][:25],
        "seasonality": _deep_clean(getattr(bundle, "seasonality", {}) or {}, depth=1),
        "theory": _deep_clean(getattr(bundle, "theory", {}) or {}, depth=2),
        "financials_tail": _frame_tail(getattr(bundle, "financials", None), 8),
        "sector_boards": (getattr(bundle, "sentiment", {}) or {}).get("boards_raw") or [],
        "stages": stages[-6:],
    }
    # 图表所需的最小行情数据（只留最近 320 根，控制缓存体积）
    df = getattr(bundle, "ohlcv", None)
    if isinstance(df, pd.DataFrame) and not df.empty:
        payload["ohlcv"] = df.tail(320).to_dict("list")
    idx = getattr(bundle, "index_data", None)
    if isinstance(idx, pd.DataFrame) and not idx.empty:
        payload["index"] = idx.tail(320)[["date", "close"]].to_dict("list")
    # 图表：以 PNG 字节形式缓存，避免在缓存里放 Figure 对象
    payload["charts"] = _render_charts_png(result, df)
    return payload


def _clean_dict(d: dict, depth: int = 1) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)) and depth > 0:
            out[k] = [x for x in v if isinstance(x, (str, int, float, bool))][:20]
        elif isinstance(v, dict) and depth > 0:
            out[k] = _clean_dict(v, depth - 1)
    return out


def _deep_clean(d, depth: int = 2):
    if depth <= 0:
        return None if isinstance(d, (dict, list)) else d
    if isinstance(d, dict):
        return {k: _deep_clean(v, depth - 1) for k, v in d.items()
                if not isinstance(v, pd.DataFrame)}
    if isinstance(d, (list, tuple)):
        return [_deep_clean(x, depth - 1) for x in d[:20]]
    if isinstance(d, (str, int, float, bool)) or d is None:
        return d
    return str(d)[:200]


def _frame_tail(df, n: int):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    keep = [c for c in ("报告期", "营业收入", "营收同比", "净利润", "净利同比", "ROE",
                        "毛利率", "净利率", "资产负债率", "经营现金流", "每股收益")
            if c in df.columns]
    if not keep:
        return None
    return df[keep].tail(n).to_dict("list")


def _capital_brief(cap: dict) -> dict:
    out: dict = {}
    flow = cap.get("flow")
    if isinstance(flow, pd.DataFrame) and not flow.empty:
        try:
            if "main_net" in flow.columns:
                out["main_net_5d"] = float(flow["main_net"].tail(5).sum())
                out["main_net_20d"] = float(flow["main_net"].tail(20).sum())
            if "date" in flow.columns:
                out["flow_last_date"] = str(flow["date"].iloc[-1])[:10]
            if len(flow) >= 2 and "main_net" in flow.columns:
                out["main_net_last"] = float(flow["main_net"].iloc[-1])
        except Exception:  # noqa: BLE001
            pass
    for key, src in (("main_force_profile", cap.get("main_force_profile") or cap.get("profile")),
                     ("control_degree", (cap.get("control_degree") or {})
                      if isinstance(cap.get("control_degree"), dict) else None)):
        if isinstance(src, dict):
            out[key] = _clean_dict(src, depth=1)
    holders = cap.get("holders")
    if isinstance(holders, pd.DataFrame) and not holders.empty:
        out["holders_tail"] = holders.tail(4).to_dict("list")
    margin = cap.get("margin")
    if isinstance(margin, pd.DataFrame) and not margin.empty:
        out["margin_tail"] = margin.tail(4).to_dict("list")
    lhb = cap.get("lhb")
    if isinstance(lhb, pd.DataFrame) and not lhb.empty:
        out["lhb"] = lhb.tail(4).to_dict("list")
    north = cap.get("north")
    if isinstance(north, pd.DataFrame) and not north.empty:
        out["north_tail"] = north.tail(4).to_dict("list")
    return out


def _render_charts_png(result, df) -> dict:
    """把图表渲染成 PNG 字节（可直接 st.image 显示，且可安全缓存）。"""
    out: dict[str, bytes] = {}
    if not isinstance(df, pd.DataFrame) or df.empty:
        return out

    def _cap(fig, quality: int = 88) -> bytes | None:
        if fig is None:
            return None
        try:
            import io

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:  # noqa: BLE001
                pass
            return buf.getvalue()
        except Exception as exc:  # noqa: BLE001
            log.debug("图表转 PNG 失败: %s", exc)
            return None

    sr = None
    try:
        from core import indicators as ind

        if hasattr(ind, "support_resistance"):
            sr = ind.support_resistance(df["high"], df["low"], df["close"],
                                        volume=df.get("volume"))
    except Exception:  # noqa: BLE001
        sr = None

    jobs = {
        "price": lambda: charts_mod.plot_price(
            df, bars=180, title="%s K线走势" % (result.name or result.symbol),
            support_resistance=sr),
        "indicators": lambda: charts_mod.plot_indicators(df, bars=180),
        "multi_horizon": lambda: charts_mod.plot_multi_horizon(result),
        "cone": lambda: charts_mod.plot_confidence_cone(result),
        "gauge": lambda: charts_mod.plot_probability_gauge(result),
        "contribution": lambda: charts_mod.plot_factor_contribution(result, "d20"),
    }
    for key, fn in jobs.items():
        try:
            data = _cap(fn())
            if data:
                out[key] = data
        except Exception as exc:  # noqa: BLE001
            log.debug("生成图表 %s 失败: %s", key, exc)
    return out


# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
POPULAR = ["600519 贵州茅台", "000001 平安银行", "300750 宁德时代", "002594 比亚迪",
           "601318 中国平安", "688981 中芯国际", "000858 五粮液", "600036 招商银行",
           "601899 紫金矿业", "002415 海康威视", "300059 东方财富", "601012 隆基绿能"]


def sidebar() -> dict:
    st.sidebar.title("📈 个股涨跌概率预测")
    st.sidebar.caption("core v%s ｜ 8 个周期 ｜ 8 大维度因子" % CORE_VERSION)

    if "symbol_input" not in st.session_state:
        st.session_state["symbol_input"] = "600519"

    symbol = st.sidebar.text_input("股票代码 / 名称", key="symbol_input",
                                   help="支持 600519、sh600519、000001.SZ、00700.HK、AAPL")
    if st.sidebar.button("🎯 开始分析", type="primary", use_container_width=True):
        st.session_state["run_analysis"] = True
        cached_analysis.clear()

    with st.sidebar.expander("常用股票", expanded=False):
        cols = st.columns(2)
        for i, item in enumerate(POPULAR):
            code = item.split()[0]
            if cols[i % 2].button(code, key="pop_%s" % code, use_container_width=True):
                st.session_state["symbol_input"] = code
                st.session_state["run_analysis"] = True
                st.rerun()

    st.sidebar.markdown("---")
    with st.sidebar.expander("分析设置", expanded=False):
        use_ml = st.checkbox("启用 AI 模型（较慢，但更全面）", value=True,
                             help="用该股历史数据训练逻辑回归/GBDT/历史类比模型，"
                                  "做扩展窗口前向验证后与规则模型融合")
        demo = st.checkbox("演示模式（用合成数据，不联网）", value=False,
                           help="无网络或数据源不可用时，用合成数据验证流程与界面")
        days = st.slider("历史数据天数", 400, 3200, 1600, step=200,
                         help="回测与 AI 训练都依赖历史长度，长周期预测建议 ≥1600")
        do_backtest = st.checkbox("执行历史回测（校准置信度）", value=True,
                                  help="用历史数据检验规则分与未来收益的相关性，"
                                       "并据此对每个周期的置信度打分")
    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 清空数据缓存", use_container_width=True):
        n = cache_clear()
        cached_analysis.clear()
        st.sidebar.success("已清理 %d 个缓存文件" % n)
    with st.sidebar.expander("关于 / 方法说明"):
        st.markdown("""
**预测流程**

1. 取数（日/周/月线、实时快照、大盘、资金流、财务、新闻…）
2. 八大维度分析，每个模块只输出统一的"因子"（方向/强度/可信度/理论依据）
3. 按 **周期权重矩阵** 汇总因子：短线重资金与技术，长线重基本面与宏观
4. 方向分 → logit 概率；波动率 × √t → 幅度区间
5. AI 模型（逻辑回归 + 梯度提升 + 历史相似形态类比）按周期融合
6. 多周期一致性检查 + 历史回测校准置信度

**重要**：模型输出的是**概率分布**，不是确定答案。
""")
    return {"symbol": symbol, "use_ml": use_ml, "demo": demo, "days": days,
            "backtest": do_backtest}


# --------------------------------------------------------------------------- #
# 渲染组件
# --------------------------------------------------------------------------- #
def pcard(label: str, value: str, sub: str = "", cls: str = "") -> str:
    return ("<div class='pcard'><div class='lbl'>%s</div>"
            "<div class='val %s'>%s</div><div class='sub'>%s</div></div>"
            % (label, cls, value, sub))


def render_header(p: dict) -> None:
    name = p.get("name") or p.get("symbol")
    st.markdown("### %s（%s）" % (name, p["symbol"]))
    bits = ["数据截止 **%s**" % (p.get("as_of") or "—"),
            "分析耗时 **%.1fs**" % p.get("elapsed", 0),
            "因子 **%d** 个" % p.get("n_factors", 0)]
    st.caption(" ｜ ".join(bits))
    if p.get("is_demo"):
        st.markdown("<div class='note'>⚠ <b>演示模式</b>：当前使用合成数据，"
                    "仅用于验证流程与界面，<b>不可用于实盘决策</b>。</div>",
                    unsafe_allow_html=True)


def render_overview(p: dict) -> None:
    price = p.get("price")
    ens = p.get("ensemble") or {}
    f20 = _f(p, "d20") or (_f(p, "d1") or {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("现价", "%.2f" % price if price else "—")
    c2.metric("综合评分", "%+.1f" % ens.get("score_pct", 0.0),
              help="8 个周期按重要性加权后的方向分，范围 -100~+100")
    c3.metric("操作倾向", ens.get("stance", "—"))
    if f20:
        c4.metric("未来20天上涨概率", "%.1f%%" % (f20["p_up"] * 100),
                  delta="%+.2f%% 期望幅度" % f20["exp_return"])

    cons = ens.get("consistency") or {}
    if cons.get("detail"):
        st.info(cons["detail"])

    cards = []
    for f in p.get("forecasts", []):
        cls = "up" if f["p_up"] >= 0.5 else "down"
        cards.append(pcard(
            f["label"], "%.1f%%" % (f["p_up"] * 100), cls=cls,
            sub="%s ｜ %+.2f%%<br>区间 %+.1f%%~%+.1f%% ｜ 置信 %.0f%%" % (
                f["trend"], f["exp_return"], f["ci_low"], f["ci_high"],
                f["confidence"] * 100)))
    st.markdown("<div class='bigprobs'>%s</div>" % "".join(cards), unsafe_allow_html=True)


def _f(p: dict, key: str):
    for f in p.get("forecasts", []):
        if f["horizon"] == key:
            return f
    return None


def render_forecast_table(p: dict) -> None:
    rows = []
    for f in p.get("forecasts", []):
        rows.append({
            "周期": f["label"], "结论": f["trend"],
            "上涨概率": "%.1f%%" % (f["p_up"] * 100),
            "下跌概率": "%.1f%%" % (f["p_down"] * 100),
            "期望涨跌幅": "%+.2f%%" % f["exp_return"],
            "乐观(80%)": "%+.2f%%" % f["exp_high"],
            "悲观(20%)": "%+.2f%%" % f["exp_low"],
            "约72%区间": "%+.1f%% ~ %+.1f%%" % (f["ci_low"], f["ci_high"]),
            "大涨概率": "%.1f%%" % (f["p_big_up"] * 100),
            "大跌概率": "%.1f%%" % (f["p_big_down"] * 100),
            "横盘概率": "%.1f%%" % (f["p_flat"] * 100),
            "波动率σ": "%.2f%%" % f["sigma"],
            "趋势强度": "%+.0f" % f["strength_score"],
            "置信度": "%.0f%%" % (f["confidence"] * 100),
            "AI参与": "%.0f%%" % (f["model_mix"].get("AI模型", 0) * 100),
        })
    st.markdown("<div class='scrollx'>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_charts(p: dict) -> None:
    ch = p.get("charts") or {}
    if not ch:
        st.info("暂无图表（数据不足）。")
        return
    tab_names = [("price", "K线与均线"), ("indicators", "技术指标"),
                 ("multi_horizon", "周期对比"), ("gauge", "概率仪表"),
                 ("cone", "置信锥"), ("contribution", "因子贡献")]
    tabs = st.tabs([n for _, n in tab_names if _ in ch])
    i = 0
    for key, label in tab_names:
        if key not in ch:
            continue
        with tabs[i]:
            st.image(ch[key], use_container_width=True)
        i += 1
    st.caption("红=上涨/看涨，绿=下跌/看跌（A 股习惯）。阴影为布林带，虚线为支撑/压力位。")


def render_drivers(p: dict) -> None:
    key = st.selectbox("选择周期", [f["horizon"] for f in p.get("forecasts", [])],
                       format_func=lambda k: (_f(p, k) or {}).get("label", k),
                       key="driver_h")
    f = _f(p, key)
    if not f:
        return
    if f.get("note"):
        st.caption("注：%s" % f["note"])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**✅ 正向贡献（看涨）**")
        for d in f.get("drivers", [])[:8]:
            st.markdown("<span class='pill bull'>%s</span> **%s** ｜ %s "
                        "<span style='color:#888;font-size:11.5px'>[%s] 强度%.2f 可信度%.2f"
                        " 贡献%+.1f%%</span>" % (
                            _esc(d.get("类别", "")), _esc(d.get("因子", "")),
                            _esc(d.get("解读", "")), _esc(d.get("理论依据", "")),
                            float(d.get("强度", 0)), float(d.get("可信度", 0)),
                            float(d.get("贡献", 0))), unsafe_allow_html=True)
    with c2:
        st.markdown("**⛔ 反向贡献（看跌/风险）**")
        for d in f.get("counter", [])[:8]:
            st.markdown("<span class='pill bear'>%s</span> **%s** ｜ %s "
                        "<span style='color:#888;font-size:11.5px'>[%s] 强度%.2f 可信度%.2f"
                        " 贡献%+.1f%%</span>" % (
                            _esc(d.get("类别", "")), _esc(d.get("因子", "")),
                            _esc(d.get("解读", "")), _esc(d.get("理论依据", "")),
                            float(d.get("强度", 0)), float(d.get("可信度", 0)),
                            float(d.get("贡献", 0))), unsafe_allow_html=True)
    ens = p.get("ensemble") or {}
    if ens.get("key_levels"):
        st.markdown("**关键价位参考**")
        cols = st.columns(3)
        for i, (k, v) in enumerate(ens["key_levels"].items()):
            cols[i % 3].metric(k, "%.2f" % float(v))
    if f.get("drivers") or f.get("counter"):
        st.markdown("**全部因子明细**")
        rows = []
        for d in list(f.get("drivers", [])) + list(f.get("counter", [])):
            rows.append({"因子": d.get("因子"), "类别": d.get("类别"), "方向": d.get("方向"),
                         "强度": d.get("强度"), "可信度": d.get("可信度"),
                         "贡献%": d.get("贡献"), "解读": d.get("解读"),
                         "理论依据": d.get("理论依据"), "观测值": str(d.get("观测值"))[:40]})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _esc(x) -> str:
    import html

    return html.escape(str(x if x is not None else ""))


def render_dimensions(p: dict) -> None:
    snap = p.get("snapshot") or {}
    stage = p.get("stage") or {}
    val = p.get("valuation") or {}
    cap = p.get("capital_brief") or {}
    sent = p.get("sentiment") or {}
    mac = p.get("macro") or {}
    sea = p.get("seasonality") or {}
    th = p.get("theory") or {}
    stats = p.get("factor_stats") or {}
    vol = p.get("volatility") or {}

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        ["① 个股特性/阶段", "② 技术面", "③ 基本面", "④ 资金面/主力", "⑤ 情绪/博弈",
         "⑥ 宏观/行业", "⑦ 消息/事件", "⑧ 周期/季节性", "⑨ 理论模型"])

    with tab1:
        _kv_grid({
            "名称": snap.get("name"), "现价": snap.get("price"),
            "涨跌幅%": snap.get("pct_chg"), "换手率%": snap.get("turnover"),
            "量比": snap.get("volume_ratio"), "振幅%": snap.get("amplitude"),
            "PE(TTM)": snap.get("pe_ttm"), "PB": snap.get("pb"),
            "总市值": _big(snap.get("market_cap")), "流通市值": _big(snap.get("float_market_cap")),
            "距涨停%": snap.get("vs_limit_up_pct"), "数据来源": snap.get("source"),
        })
        if stage:
            st.markdown("**所处阶段与特性判定**")
            _kv_grid({k: _short(v) for k, v in stage.items()})
        if vol:
            st.markdown("**波动率结构**")
            _kv_grid({"日波动率%": _r(vol.get("daily_pct"), 3),
                      "年化波动率%": _r(vol.get("annual_pct"), 2),
                      "20日波动%": _r(vol.get("vol_20"), 3),
                      "60日波动%": _r(vol.get("vol_60"), 3),
                      "短长波动比": _r(vol.get("vol_ratio"), 3),
                      "波动状态": vol.get("regime"), "ATR%": _r(vol.get("atr_pct"), 2)})

    with tab2:
        st.markdown("技术面因子已计入上面的因子明细。下面是关键指标的当前读数：")
        rows = [d for d in _all_factors(p) if d.get("类别") == "技术面"]
        if rows:
            st.dataframe(pd.DataFrame([{
                "因子": d.get("因子"), "方向": d.get("方向"), "强度": d.get("强度"),
                "可信度": d.get("可信度"), "观测值": str(d.get("观测值"))[:50],
                "解读": d.get("解读"), "理论依据": d.get("理论依据"),
            } for d in rows]), use_container_width=True, hide_index=True)
        else:
            st.info("未取得技术面因子。")

    with tab3:
        if val:
            _kv_grid({"PE(TTM)": _r(val.get("pe_ttm")), "PB": _r(val.get("pb")),
                      "PE历史分位": _pct(val.get("pe_percentile")),
                      "PB历史分位": _pct(val.get("pb_percentile")),
                      "总市值": _big(val.get("market_cap")),
                      "股息率%": _r(val.get("dividend_yield")),
                      "估值来源": val.get("source")})
            if val.get("percentile_method"):
                st.caption("估值分位口径：%s" % val["percentile_method"])
        fin = p.get("financials_tail")
        if fin:
            st.markdown("**财务摘要（最近若干期）**")
            st.dataframe(pd.DataFrame(fin), use_container_width=True, hide_index=True)
        else:
            st.info("未取得财务数据。")

    with tab4:
        _kv_grid({"近5日主力净额": _big(cap.get("main_net_5d")),
                  "近20日主力净额": _big(cap.get("main_net_20d")),
                  "最近一日主力净额": _big(cap.get("main_net_last")),
                  "资金流数据日": cap.get("flow_last_date")})
        prof = cap.get("main_force_profile") or {}
        if prof:
            st.markdown("**主力行为判定**")
            _kv_grid({k: _short(v) for k, v in prof.items()})
        cd = cap.get("control_degree") or {}
        if cd:
            st.markdown("**控盘程度**")
            _kv_grid({k: _short(v) for k, v in cd.items()})
        for key, label in (("holders_tail", "股东户数（户均持股）"),
                           ("margin_tail", "融资融券"), ("lhb", "龙虎榜"),
                           ("north_tail", "北向持股")):
            v = cap.get(key)
            if v:
                st.markdown("**%s**" % label)
                st.dataframe(pd.DataFrame(v), use_container_width=True, hide_index=True)

    with tab5:
        if sent:
            for k, v in sent.items():
                if isinstance(v, dict) and v:
                    st.markdown("**%s**" % k)
                    _kv_grid({kk: _short(vv) for kk, vv in v.items()})
                elif isinstance(v, (str, int, float, bool)) and v is not None:
                    st.write("%s：%s" % (k, v))
        else:
            st.info("未取得情绪数据。")
        boards = p.get("sector_boards") or []
        if boards:
            st.markdown("**所属板块**")
            st.dataframe(pd.DataFrame(boards), use_container_width=True, hide_index=True)

    with tab6:
        if mac:
            for k, v in mac.items():
                if isinstance(v, dict) and v:
                    st.markdown("**%s**" % k)
                    _kv_grid({kk: _short(vv) for kk, vv in v.items()})
        else:
            st.info("未取得宏观数据。")

    with tab7:
        news = p.get("news") or []
        if news:
            rows = []
            for n in news:
                d = n.get("direction", 0)
                rows.append({"日期": str(n.get("date", ""))[:10],
                             "标题": str(n.get("title", ""))[:70],
                             "标签": n.get("tag", ""),
                             "倾向": "利好" if d and d > 0 else ("利空" if d and d < 0 else "中性"),
                             "来源": n.get("source", "")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("未取得近期新闻/公告。")

    with tab8:
        if sea:
            flat = {k: _short(v) for k, v in sea.items() if not isinstance(v, (dict, list))}
            _kv_grid(flat)
            for k, v in sea.items():
                if isinstance(v, dict) and v:
                    st.markdown("**%s**" % k)
                    _kv_grid({kk: _short(vv) for kk, vv in v.items()})
        else:
            st.info("未取得季节性数据。")

    with tab9:
        if th:
            labels = {"dow": "道氏理论", "elliott": "艾略特波浪", "fib": "斐波那契",
                      "gann": "江恩理论", "vpt": "量价时空", "fan": "扇形原理",
                      "pct": "百分比回撤", "chaos": "混沌理论", "resonance": "多周期共振",
                      "consensus": "理论综合"}
            for k, v in th.items():
                if isinstance(v, dict) and v:
                    st.markdown("**%s**" % labels.get(k, k))
                    _kv_grid({kk: _short(vv) for kk, vv in v.items()})
                elif isinstance(v, (str, int, float, bool)) and v is not None:
                    st.write("%s：%s" % (k, v))
        else:
            st.info("未取得理论模型结果。")

    if stats.get("by_category"):
        st.markdown("---")
        st.markdown("**因子分布**")
        rows = [{"类别": k, "看涨": v.get("看涨", 0), "看跌": v.get("看跌", 0),
                 "中性": v.get("中性", 0), "合计": sum(v.values())}
                for k, v in stats["by_category"].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _all_factors(p: dict) -> list:
    seen, out = set(), []
    for f in p.get("forecasts", []):
        for d in list(f.get("drivers", [])) + list(f.get("counter", [])):
            k = (d.get("因子"), str(d.get("观测值")))
            if k in seen:
                continue
            seen.add(k)
            out.append(d)
    return out


def _kv_grid(d: dict) -> None:
    items = [(k, v) for k, v in d.items()
             if v is not None and v == v and not isinstance(v, (dict, list))]
    if not items:
        st.caption("（无数据）")
        return
    html = ["<div class='bigprobs'>"]
    for k, v in items:
        html.append("<div class='pcard'><div class='lbl'>%s</div>"
                    "<div class='val' style='font-size:15px'>%s</div></div>"
                    % (_esc(k), _esc(_short(v))))
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _short(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        if v != v:
            return "—"
        return "%.4g" % v
    s = str(v)
    return s[:79] + "…" if len(s) > 80 else s


def _r(v, n: int = 2) -> str:
    try:
        f = float(v)
        return "—" if f != f else ("%%.%df" % n) % f
    except (TypeError, ValueError):
        return "—"


def _pct(v) -> str:
    try:
        f = float(v)
        return "—" if f != f else "%.1f%%" % (f * 100)
    except (TypeError, ValueError):
        return "—"


def _big(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if f != f:
        return "—"
    for unit, div in (("万亿", 1e12), ("亿", 1e8), ("万", 1e4)):
        if abs(f) >= div:
            return "%.2f%s" % (f / div, unit)
    return "%.2f" % f


def render_backtest(p: dict, do_backtest: bool, symbol: str, days: int, demo: bool) -> None:
    if not do_backtest:
        st.info("已在侧边栏关闭历史回测。")
        return
    with st.spinner("正在用历史数据回测并校准置信度…"):
        try:
            df = pd.DataFrame(p.get("ohlcv") or {})
            if df.empty:
                st.warning("缺少行情数据，无法回测。")
                return
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            idx = pd.DataFrame(p.get("index") or {})
            res = st.session_state.get("_bt_cache_%s" % symbol)
            if res is None:
                bt = bt_mod.backtest_prediction_history(df, idx if not idx.empty else None)
                tbl = bt_mod.summarize(bt)
                grade = bt_mod.reliability_grade(bt)
                res = {"bt": bt, "table": tbl, "grade": grade}
                st.session_state["_bt_cache_%s" % symbol] = res
            tbl = res["table"]
            grade = res["grade"]
            st.markdown("**可信度评级：%s**（评分 %.2f）" % (grade.get("grade"), grade.get("score", 0)))
            st.caption(grade.get("detail", ""))
            if not tbl.empty:
                st.markdown("<div class='scrollx'>", unsafe_allow_html=True)
                st.dataframe(tbl, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
            fig = charts_mod.plot_backtest(tbl, grade)
            if fig is not None:
                import io

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
                st.image(buf.getvalue(), use_container_width=True)
                try:
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                except Exception:  # noqa: BLE001
                    pass
            st.caption("IC>0.03 视为信号有效；IC≈0 表示该周期的历史规律很弱，"
                       "置信度会被自动下调。样本数越少，结论越不可靠。")
        except Exception as exc:  # noqa: BLE001
            st.warning("回测失败：%s" % exc)


def render_export(p: dict, symbol: str) -> None:
    """导出：优先用完整版报告生成器（与本地版完全一致），失败时退回简版。"""
    st.markdown("**下载报告**")

    @st.cache_data(show_spinner=False, ttl=1800, max_entries=8)
    def _build_full_report(sym: str, payload_json: str, kind: str) -> str:
        """用 report 模块生成与本地版一致的完整 HTML / 文本报告。

        为了能被 Streamlit 缓存，这里传入的是 payload 的 JSON 字符串
        （payload 本身含 DataFrame 转换结果，未必可哈希）。
        """
        import json as _json

        pl = _json.loads(payload_json)
        res = _payload_to_result(pl)
        if res is None:
            raise RuntimeError("无法重建结果对象")
        if kind == "html":
            return report_mod.build_html(res)
        return report_mod.build_text(res)

    html_txt = txt_txt = None
    try:
        import json as _json

        pj = _json.dumps(p, ensure_ascii=False, default=str)
        html_txt = _build_full_report(symbol, pj, "html")
        txt_txt = _build_full_report(symbol, pj, "txt")
    except Exception as exc:  # noqa: BLE001
        log.debug("完整报告生成失败，退回简版: %s", exc)
        try:
            html_txt = _payload_html(p)
            txt_txt = _payload_text(p)
        except Exception:  # noqa: BLE001
            pass

    c1, c2 = st.columns(2)
    if html_txt:
        c1.download_button("⬇️ 下载 HTML 报告", html_txt.encode("utf-8"),
                           file_name="%s_预测报告.html" % symbol, mime="text/html",
                           use_container_width=True)
    if txt_txt:
        c2.download_button("⬇️ 下载文本报告", txt_txt.encode("utf-8"),
                           file_name="%s_预测报告.txt" % symbol, mime="text/plain",
                           use_container_width=True)
    if not html_txt and not txt_txt:
        st.caption("报告生成失败。")
    else:
        st.caption("报告内容与本地版 `python run_local.py %s` 生成的一致。" % symbol)


def _payload_to_result(p: dict):
    """把缓存 payload 还原成 ``PredictionResult``（用于完整报告生成）。"""
    try:
        from core.contracts import AnalysisBundle, ProbabilityForecast, PredictionResult

        fcs = []
        for f in p.get("forecasts", []):
            fcs.append(ProbabilityForecast(
                horizon=f["horizon"], label=f["label"], bars=0,
                p_up=f["p_up"], p_down=f.get("p_down", 1 - f["p_up"]),
                p_flat=f.get("p_flat", 0.0), p_big_up=f.get("p_big_up", 0.0),
                p_big_down=f.get("p_big_down", 0.0), exp_return=f["exp_return"],
                exp_return_low=f.get("exp_low", 0.0), exp_return_high=f.get("exp_high", 0.0),
                ci_low=f["ci_low"], ci_high=f["ci_high"], sigma=f.get("sigma", 0.0),
                big_move_threshold=f.get("big_move", 5.0), trend=f["trend"],
                confidence=f["confidence"], strength_score=f.get("strength_score", 0.0),
                drivers=list(f.get("drivers") or []), counter=list(f.get("counter") or []),
                model_mix=dict(f.get("model_mix") or {}), note=f.get("note", "")))
        bundle = AnalysisBundle(
            symbol=p.get("symbol", ""), name=p.get("name", ""), as_of=p.get("as_of", ""),
            notes=list(p.get("notes") or []), errors=list(p.get("meta_errors") or []),
            snapshot=dict(p.get("snapshot") or {}), stage=dict(p.get("stage") or {}),
            valuation=dict(p.get("valuation") or {}),
            sentiment=dict(p.get("sentiment") or {}), macro=dict(p.get("macro") or {}),
            seasonality=dict(p.get("seasonality") or {}), theory=dict(p.get("theory") or {}),
            news=list(p.get("news") or []),
            indicators={"volatility": p.get("volatility") or {},
                        "cross_horizon": (p.get("ensemble") or {}).get("consistency") or {}})
        return PredictionResult(
            symbol=p.get("symbol", ""), name=p.get("name", ""), as_of=p.get("as_of", ""),
            price=float(p.get("price") or float("nan")), forecasts=fcs, bundle=bundle,
            ensemble=p.get("ensemble") or {},
            meta={"sources": p.get("meta_sources") or {}, "errors": p.get("meta_errors") or [],
                  "notes": p.get("notes") or [], "n_factors": p.get("n_factors")})
    except Exception as exc:  # noqa: BLE001
        log.debug("payload -> result 转换失败: %s", exc)
        return None


def render_data_health() -> None:
    if st.button("🩺 数据源健康检查", use_container_width=False):
        with st.spinner("正在并发检测各数据源…"):
            try:
                from core import datasource as ds

                rep = ds.data_health_check()
                st.json(rep)
            except Exception as exc:  # noqa: BLE001
                st.warning("健康检查不可用：%s" % exc)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    opts = sidebar()

    if CORE_ERROR:
        st.error("核心库 core 导入失败，请检查目录结构（app.py 应与 core/ 同级）。")
        with st.expander("错误详情"):
            st.code(CORE_ERROR)
        return

    st.markdown("## 个股涨跌概率预测")
    st.caption("技术面 · 基本面 · 资金面 · 情绪面 · 宏观 · 消息面 · 周期 · 金融理论 → "
               "未来 1天/5天/10天/20天/1月/1季/半年/1年 的涨跌概率与幅度推断")

    sym_raw = (opts["symbol"] or "").strip()
    if not sym_raw:
        st.info("请在左侧输入股票代码，例如 `600519`、`000001`、`300750`。")
        st.markdown("<div class='risk'>%s</div>" % DISCLAIMER, unsafe_allow_html=True)
        return

    if not st.session_state.get("run_analysis"):
        st.info("👈 在左侧输入代码后点击 **开始分析**。")
        st.markdown("<div class='risk'>%s</div>" % DISCLAIMER, unsafe_allow_html=True)
        return

    prog = st.progress(0.0, text="准备中…")
    try:
        prog.progress(0.15, text="获取行情与八大维度数据（首次约需 10~60 秒）…")
        payload = cached_analysis(sym_raw, bool(opts["use_ml"]), bool(opts["demo"]),
                                  int(opts["days"]))
        prog.progress(1.0, text="完成")
        time.sleep(0.15)
        prog.empty()
    except Exception as exc:  # noqa: BLE001
        prog.empty()
        st.error("分析失败：%s" % exc)
        with st.expander("错误详情"):
            st.code(traceback.format_exc())
        st.info("提示：可勾选侧边栏的 **演示模式** 用合成数据验证界面，"
                "或点击 **数据源健康检查** 排查网络问题。")
        return

    symbol = payload.get("symbol") or sym_raw
    render_header(payload)
    render_overview(payload)

    tabs = st.tabs(["📊 预测总览", "📈 图表", "🔍 因子与逻辑",
                    "🧭 八维明细", "🧪 历史验证", "📥 导出/工具"])

    with tabs[0]:
        render_forecast_table(payload)
        st.caption("说明：**大涨/大跌概率**按各周期的阈值计算（如1天 ±3%、20天 ±12%、1年 ±50%）；"
                   "**置信度**已根据历史回测结果自动折扣。")
    with tabs[1]:
        render_charts(payload)
    with tabs[2]:
        render_drivers(payload)
    with tabs[3]:
        render_dimensions(payload)
    with tabs[4]:
        render_backtest(payload, bool(opts["backtest"]), symbol, int(opts["days"]),
                        bool(opts["demo"]))
    with tabs[5]:
        render_export(payload, symbol)
        st.markdown("---")
        render_data_health()
        if payload.get("meta_errors"):
            st.markdown("**未能完成的模块**")
            for e in payload["meta_errors"]:
                st.caption("⚠ %s" % e)
        if payload.get("notes"):
            st.markdown("**数据说明**")
            for n in payload["notes"]:
                st.caption("· %s" % n)
        st.caption("数据来源：%s" % payload.get("meta_sources"))

    st.markdown("<div class='risk'>%s</div>" % DISCLAIMER, unsafe_allow_html=True)


def _payload_text(p: dict) -> str:
    """不依赖完整 result 对象的简版文本报告（用于下载）。"""
    L = ["=" * 84,
         "%s（%s） 涨跌概率预测报告" % (p.get("name") or "", p.get("symbol") or ""),
         "数据截止 %s ｜ 现价 %s" % (p.get("as_of") or "—", p.get("price")),
         "=" * 84, ""]
    ens = p.get("ensemble") or {}
    L.append("综合评分 %+.1f/100 ｜ 操作倾向：%s" % (ens.get("score_pct", 0),
                                                    ens.get("stance", "—")))
    cons = ens.get("consistency") or {}
    if cons.get("detail"):
        L.append(cons["detail"])
    L.append("")
    L.append("%-18s %-6s %-9s %-9s %-11s %-22s %-8s" % (
        "周期", "结论", "上涨概率", "下跌概率", "期望涨跌幅", "约72%区间", "置信度"))
    for f in p.get("forecasts", []):
        L.append("%-18s %-6s %8.1f%% %8.1f%% %+10.2f%% %10s %7.0f%%" % (
            f["label"], f["trend"], f["p_up"] * 100, f["p_down"] * 100, f["exp_return"],
            "%+.1f%%~%+.1f%%" % (f["ci_low"], f["ci_high"]), f["confidence"] * 100))
    L.append("")
    if ens.get("top_drivers"):
        L.append("主要驱动：" + "、".join(d["因子"] for d in ens["top_drivers"]))
    if ens.get("top_risks"):
        L.append("主要风险：" + "、".join(d["因子"] for d in ens["top_risks"]))
    if ens.get("key_levels"):
        L.append("")
        L.append("关键价位（未来20天）：")
        for k, v in ens["key_levels"].items():
            L.append("  %-14s %.2f" % (k, float(v)))
    f20 = None
    for f in p.get("forecasts", []):
        if f["horizon"] == "d20":
            f20 = f
    if f20:
        L.append("")
        L.append("未来20天因子明细：")
        for d in f20.get("drivers", [])[:10]:
            L.append("  + %-20s %s（贡献%+.1f%%）" % (d.get("因子"), d.get("解读"),
                                                     float(d.get("贡献", 0))))
        for d in f20.get("counter", [])[:10]:
            L.append("  - %-20s %s（贡献%+.1f%%）" % (d.get("因子"), d.get("解读"),
                                                     float(d.get("贡献", 0))))
    L.append("")
    L.append("免责声明：本报告由量化模型自动生成，概率与幅度均为统计推断，不构成投资建议。")
    return "\n".join(L)


def _payload_html(p: dict) -> str:
    import html as H

    body = ["<h1>%s（%s）涨跌概率预测报告</h1>" % (H.escape(str(p.get("name") or "")),
                                                 H.escape(str(p.get("symbol") or ""))),
            "<p>数据截止 %s ｜ 现价 %s</p>" % (p.get("as_of") or "—", p.get("price"))]
    ens = p.get("ensemble") or {}
    body.append("<h2>综合结论</h2><p>评分 %+.1f/100 ｜ 倾向：%s</p>" % (
        ens.get("score_pct", 0), H.escape(str(ens.get("stance", "—")))))
    if (ens.get("consistency") or {}).get("detail"):
        body.append("<p>%s</p>" % H.escape(ens["consistency"]["detail"]))
    body.append("<h2>八大周期</h2><table border='1' cellspacing='0' cellpadding='5'>"
                "<tr><th>周期</th><th>结论</th><th>上涨概率</th><th>下跌概率</th>"
                "<th>期望涨跌幅</th><th>区间</th><th>置信度</th></tr>")
    for f in p.get("forecasts", []):
        body.append("<tr><td>%s</td><td>%s</td><td>%.1f%%</td><td>%.1f%%</td>"
                    "<td>%+.2f%%</td><td>%+.1f%%~%+.1f%%</td><td>%.0f%%</td></tr>" % (
                        H.escape(f["label"]), H.escape(f["trend"]), f["p_up"] * 100,
                        f["p_down"] * 100, f["exp_return"], f["ci_low"], f["ci_high"],
                        f["confidence"] * 100))
    body.append("</table>")
    if ens.get("key_levels"):
        body.append("<h2>关键价位</h2><ul>")
        for k, v in ens["key_levels"].items():
            body.append("<li>%s：%.2f</li>" % (H.escape(k), float(v)))
        body.append("</ul>")
    body.append("<h2>免责声明</h2><p>%s</p>" % H.escape(DISCLAIMER.replace("**", "")))
    return ("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>预测报告</title></head><body style='font-family:sans-serif;"
            "max-width:900px;margin:0 auto;padding:16px'>" + "".join(body) +
            "</body></html>")


if __name__ == "__main__":
    main()
