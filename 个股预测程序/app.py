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
import re
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
                         "田夫 开发。仅供研究，不构成投资建议。"},
)

# --------------------------------------------------------------------------- #
# 样式（移动端友好）
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
  /* ---------- 布局：自适应宽度，避免右侧内容被裁切 ---------- */
  .block-container { padding-top: 1.0rem; padding-bottom: 2.2rem;
                     max-width: 100%; padding-left: 1.1rem; padding-right: 1.1rem; }
  @media (min-width: 1400px) { .block-container { max-width: 1400px; margin: 0 auto; } }

  div[data-testid="stMetricValue"] { font-size: 1.4rem; line-height: 1.25; }
  div[data-testid="stMetricLabel"] { font-size: .82rem; }

  /* ---------- 概率卡片 ---------- */
  .bigprobs { display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 4px; }
  .pcard { flex:1 1 118px; min-width:0; border:1px solid #e3e8ee; border-radius:9px;
           padding:9px 11px; background:#fbfcfe; }
  .pcard .lbl { font-size:12px; color:#5b6b7c; }
  .pcard .val { font-size:19px; font-weight:700; }
  .pcard .sub { font-size:11px; color:#7b8a99; line-height:1.45; }
  /* 缺失数据的卡片：灰底 + 虚框，明确区分"没这项分析"与"数据源没给" */
  .pcard.miss { background:#f6f7f9; border-style:dashed; border-color:#d9dee5; }
  .pcard.miss .lbl { color:#9aa7b4; }
  .pcard.miss .val { color:#b6c0ca; }
  .up { color:#c62828; } .down { color:#2e7d32; } .flat { color:#607d8b; }

  /* ---------- 顶部标题区 ---------- */
  .hero { border-radius:12px; padding:16px 20px 14px;
          background:linear-gradient(135deg,#12395e 0%,#1f5f8b 55%,#2f7fa8 100%);
          color:#fff; margin-bottom:12px; }
  .hero h1 { margin:0; font-size:24px; font-weight:700; color:#fff;
             letter-spacing:.5px; line-height:1.34; }
  .hero .code { font-size:14px; opacity:.9; font-weight:400; margin-left:6px; }
  .hero .meta { margin-top:8px; font-size:12.5px; opacity:.92; line-height:1.7; }
  .hero .tags { margin-top:9px; }
  .hero .tag { display:inline-block; background:rgba(255,255,255,.18);
               border:1px solid rgba(255,255,255,.28); border-radius:11px;
               padding:1px 9px; font-size:11.5px; margin:2px 5px 2px 0; }
  .hero .tag.warn { background:rgba(255,193,7,.25); border-color:rgba(255,193,7,.55); }

  /* 程序名称横幅：给足上下留白，避免标题贴住视口顶边被裁切 */
  .appbanner { padding:20px 22px 18px; margin-top:2px; scroll-margin-top:80px; }
  .appbanner h1 { font-size:25px; }
  /* 开发者署名：比正文再小 2 号（11px → 9px），弱化显示 */
  .appbanner .byline { margin-top:9px; font-size:9px; opacity:.62; letter-spacing:.3px;
                       line-height:1.5; }

  /* 个股结果卡：浅色，与深色横幅区分开 */
  .hero.stock { background:#fff; color:#1b2b3a; border:1px solid #dde5ec;
                border-left:5px solid #1f5f8b; }
  .hero.stock h1 { color:#12395e; }
  .hero.stock .meta { color:#4a5c6d; opacity:1; }
  .hero.stock .tag { background:#eef4fa; border-color:#d3e2ef; color:#284b68; }
  .hero.stock .tag.warn { background:#fff4d6; border-color:#f0cd7a; color:#8a5a00; }

  /* ---------- 表格：横向可滚动，不撑破容器 ---------- */
  div[data-testid="stDataFrame"] { max-width:100%; }
  div[data-testid="stDataFrame"] > div { overflow-x:auto !important;
                                         -webkit-overflow-scrolling:touch; }
  .scrollx { overflow-x:auto; -webkit-overflow-scrolling:touch; max-width:100%; }

  .pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11.5px;
          background:#eceff1; margin:1px 3px 1px 0; }
  .pill.bull { background:#ffebee; color:#c62828; }
  .pill.bear { background:#e8f5e9; color:#2e7d32; }
  .note { background:#fff8e1; border-left:4px solid #ffa000; padding:9px 12px;
          border-radius:5px; font-size:12.5px; margin:8px 0; }
  .risk { background:#fdecea; border-left:4px solid #c62828; padding:9px 12px;
          border-radius:5px; font-size:12.5px; margin:8px 0; }
  .tip { background:#e8f2fb; border-left:4px solid #1976d2; padding:10px 13px;
         border-radius:5px; font-size:12.8px; margin:8px 0; line-height:1.7; }

  /* ---------- 手机端 ---------- */
  @media (max-width: 700px) {
    .block-container { padding-left: .6rem; padding-right: .6rem; }
    .hero { padding:13px 14px; border-radius:10px; }
    .hero h1 { font-size:19px; }
    .hero .meta { font-size:11.5px; }
    div[data-testid="stMetricValue"] { font-size:1.12rem; }
    .pcard { flex:1 1 46%; padding:8px 9px; }
    .pcard .val { font-size:17px; }
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
def cached_analysis(symbol: str, use_ml: bool, demo: bool, days: int,
                    self_learning: bool = True, _progress_cb=None) -> dict:
    """执行一次完整分析并返回可序列化的结果字典。

    ``_progress_cb`` 前面的下划线是 **Streamlit 的约定**：告诉 cache_data
    不要尝试哈希这个参数（函数对象无法哈希/序列化，否则会直接报
    UnhashableParamError）。带下划线的参数不参与缓存键的计算，
    因此这个进度回调不会影响缓存命中。

    注意：matplotlib 图对象与 DataFrame 不放进缓存（体积大、可能不可序列化），
    只缓存**结论性数据**；图表每次由轻量函数现场生成。
    """
    t0 = time.time()
    predictor = Predictor(symbol, use_ml=use_ml, demo=demo, days=days, verbose=False,
                          self_learning=self_learning)

    stages: list = []

    def _progress(stage: str, pct: float) -> None:
        stages.append((stage, pct))
        # 把进度实时透传给界面（Streamlit 首次运行会真正更新进度条；
        # 命中缓存时不会再回调，界面直接显示完成）
        if _progress_cb is not None:
            try:
                _progress_cb(stage, pct)
            except Exception:  # noqa: BLE001
                pass

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
        # 完整因子清单（引擎里其实有 100+ 条，但每个周期只展示贡献度 top8，
        # 所以必须单独导出全量，否则八维明细里会缺 KDJ 这类因子）
        "factors_all": [_factor_row(f) for f in (getattr(bundle, "factors", []) or [])],
        "volatility": (bundle.indicators.get("volatility") if bundle is not None else {}),
        "snapshot": _clean_dict(getattr(bundle, "snapshot", {}) or {}),
        "stage": _clean_dict(getattr(bundle, "stage", {}) or {}),
        "valuation": _clean_dict(getattr(bundle, "valuation", {}) or {}),
        "capital_brief": _capital_brief(getattr(bundle, "capital", {}) or {}),
        "sentiment": _deep_clean(getattr(bundle, "sentiment", {}) or {}, depth=9),
        "macro": _deep_clean(getattr(bundle, "macro", {}) or {}, depth=9),
        "news": [n for n in (getattr(bundle, "news", []) or []) if isinstance(n, dict)][:25],
        "seasonality": _deep_clean(getattr(bundle, "seasonality", {}) or {}, depth=9),
        "theory": _deep_clean(getattr(bundle, "theory", {}) or {}, depth=9),
        "financials_tail": _frame_tail(getattr(bundle, "financials", None), 8),
        "sector_boards": (getattr(bundle, "sentiment", {}) or {}).get("boards_raw") or [],
        # ---- 大盘与外围市场（A股/港股/美股/VIX/A50）----
        "market_global": _clean_global_market(getattr(bundle, "market", {}) or {}),
        # ---- 主力操盘意图 ----
        "main_intent": _clean_dict((getattr(bundle, "capital", {}) or {}).get("main_intent")
                                   or {}, depth=2),
        "intent_series": [x for x in
                          ((getattr(bundle, "capital", {}) or {}).get("intent_series") or [])
                          if isinstance(x, dict)][-30:],
        "stages": stages[-6:],
        # ---- 自我学习 ----
        "learning": _deep_clean(result.meta.get("learning") or {}, depth=9),
        "recorded": _deep_clean(result.meta.get("recorded") or {}, depth=9),
        "learning_report": _clean_learning_report(result.symbol),
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


def _factor_row(f) -> dict:
    """把 Factor 对象转成可缓存的扁平 dict（供八维明细使用全量因子）。"""
    try:
        from core.contracts import Category
    except Exception:  # noqa: BLE001
        Category = None

    def _label(cat):
        if Category is not None:
            return Category.LABEL.get(cat, cat)
        return cat

    return {
        "因子": getattr(f, "name", ""),
        "id": getattr(f, "id", ""),
        "类别": _label(getattr(f, "category", "")),
        "方向": "看涨" if getattr(f, "direction", 0) > 0
                else ("看跌" if getattr(f, "direction", 0) < 0 else "中性"),
        "强度": round(float(getattr(f, "strength", 0) or 0), 3),
        "可信度": round(float(getattr(f, "reliability", 0) or 0), 3),
        "观测值": str(getattr(f, "value", "") or "")[:120],
        "解读": str(getattr(f, "detail", "") or "")[:220],
        "理论依据": str(getattr(f, "theory", "") or ""),
        "适用周期": ("全部" if getattr(f, "horizon", None) is None
                     else "/".join(getattr(f, "horizon"))),
    }


#: 内部技术字段 → 中文标签。值为 None 表示"不展示给用户"（数据源名、原始键名等）。
_FIELD_LABELS = {
    "ROE": "净资产收益率%",
    "_anchor": None,
    "_near_cycle": None,
    "_unit": None,
    "above_ma250": "是否在年线上方",
    "agreement": "一致度",
    "alias": "别名",
    "aligned": None,
    "amount": "成交额",
    "amp20": "20日均振幅%",
    "anchor_date": "起算日",
    "angle_label": None,
    "bars": None,
    "bars_ahead": "距今交易日",
    "bars_since": "已过交易日",
    "behavior": "主力行为",
    "bench_20d": "基准近20日%",
    "bench_5d": "基准近5日%",
    "bench_60d": "基准近60日%",
    "bench_name": "基准指数",
    "best_day": "最强星期",
    "best_day_name": "最强星期",
    "beta_250d": "Beta(250日)",
    "bid_ask": None,
    "bid_ask_unit": None,
    "big_1": "大单净额",
    "big_move": "是否大涨大跌",
    "big_move_threshold": "大涨阈值%",
    "big_net": "大单净流入",
    "big_net_last": "最近一日大单",
    "blended_ret_pct": "融合后收益%",
    "boards_raw": None,      # 板块原始表，已有更友好的展示
    "bps": "每股净资产",
    "breadth_ratio": "涨跌家数比",
    "breadth_source": None,
    "broken_rate": "炸板率",
    "bull_count": "看涨模型数",
    "by_weekday": "按星期统计",
    "calendar": "日历效应",
    "cash": None,
    "change": "变化",
    "change_5d_pct": "5日变化%",
    "chaos": "混沌理论",
    "checks": "检查项",
    "ci_high": None,
    "ci_low": None,
    "code": None,
    "commodity": "上下游商品",
    "commodity_relevant": None,
    "components": "分项",
    "computed": "统计值",
    "concordance": "一致度",
    "confidence": "置信度",
    "continuous_count": "连板家数",
    "control_degree": "控盘程度",
    "copper": "铜",
    "corr_250d": "相关性(250日)",
    "counter": None,
    "cross": None,
    "crude_oil": "原油",
    "currency": None,
    "current": "最新值",
    "current_line": "当前角度线",
    "current_month": "当前月份",
    "current_zone": "当前区域",
    "cycles": "周期列表",
    "daily": "日线",
    "date": "日期",
    "days_ahead": "距今交易日",
    "detail": "说明",
    "dimension_weights": None,
    "dir": None,
    "divergence": "背离判定",
    "divergence_detail": "背离说明",
    "dividend_yield": "股息率%",
    "dow": "道氏理论",
    "down": None,
    "drivers": None,
    "earnings": "财报窗口",
    "effects": "生效效应",
    "elliott": "艾略特波浪",
    "end": None,
    "eps_ttm": "每股收益TTM",
    "ev_ebitda": "EV/EBITDA",
    "evaluated": None,
    "evidence": None,
    "excess_20d": "近20日超额收益%",
    "excess_5d": "近5日超额收益%",
    "excess_60d": "近60日超额收益%",
    "exp_high": "乐观情形%",
    "exp_low": "悲观情形%",
    "exp_return": "期望涨跌幅%",
    "extra": None,
    "f56": None,
    "f58": None,
    "f59": None,
    "f62": None,
    "f63": None,
    "f64": None,
    "f65": None,
    "f66": None,
    "f67": None,
    "f68": None,
    "f69": None,
    "f70": None,
    "f71": None,
    "f74": None,
    "f75": None,
    "f76": None,
    "f79": None,
    "f80": None,
    "f83": None,
    "f85": None,
    "f86": None,
    "fan": "扇形原理",
    "fear_greed": "恐慌/贪婪指数",
    "fib": "斐波那契",
    "field_count": None,
    "flat_count": "平盘家数",
    "float_market_cap": "流通市值",
    "float_shares": "流通股本",
    "flow_last_date": "资金流数据日",
    "flow_rows": "资金流天数",
    "fractal_dimension": "分形维数",
    "gann": "江恩理论",
    "gann_angles": "江恩角度线",
    "gann_cycle": "江恩时间周期",
    "gann_fan": "江恩扇形线",
    "gap_days": None,
    "gdp_yoy": "GDP同比%",
    "gold": "黄金",
    "group": None,
    "guba_rank": "股吧人气排名",
    "hh_hl": "高点抬高+低点抬高",
    "high": "区间高点",
    "high_price": "最高",
    "hist_first3_ret": "次月前3日历史收益%",
    "hist_last3_ret": "月末3日历史收益%",
    "hist_month_ret_pct": "该月历史平均收益%",
    "hist_ret_pct": None,
    "history": None,            # 两融历史序列，仅作图用
    "holder_concentration": "筹码集中度",
    "holders_tail": "股东户数",
    "holiday": "长假效应",
    "horizon": None,
    "hottest": "最热板块",
    "hurst": "Hurst指数",
    "id": None,
    "idx": "序号",
    "in_session": "是否在两会期间",
    "indices": "指数",
    "industries": "命中行业",
    "industry_names": "所属行业",
    "inflow_days": "净流入天数",
    "intent": "意图",
    "intent_cn": "意图",
    "iron_ore": "铁矿石",
    "is_earnings_window": "是否财报窗口",
    "is_month_end": "是否月末",
    "is_post_holiday": "是否长假后",
    "is_pre_holiday": "是否长假前",
    "is_two_sessions": "是否两会期间",
    "items": None,
    "key": None,
    "label": None,
    "large_cap_pct": "大盘权重近20日%",
    "last_date": "数据日期",
    "last_update": "上次学习",
    "lead_corr": "领先相关",
    "level": None,
    "lhb": "龙虎榜",
    "limit_down": "跌停价",
    "limit_down_count": "跌停家数",
    "limit_up": "涨停价",
    "linkage": None,
    "lithium_carbonate": "碳酸锂",
    "lyapunov_est": "李雅普诺夫指数",
    "ma": "均线",
    "macro": "宏观经济指标",
    "main_force_profile": "主力行为",
    "main_net": "主力净流入",
    "main_net_10d": "10日主力净额",
    "main_net_20d": "近20日主力净额",
    "main_net_5d": "近5日主力净额",
    "main_net_last": "最近一日主力净额",
    "main_pct": "主力净占比%",
    "main_pct_5": "近5日主力净占比%",
    "main_pct_last": "主力净占比%",
    "margin": "两融数据",
    "margin_percentile": "两融分位",
    "margin_tail": "融资融券",
    "market": "全市场情绪",
    "market_cap": "总市值",
    "market_flag": None,
    "market_regime": "大盘水位",
    "market_state": "市场状态",
    "mean_ret_pct": "平均收益%",
    "metrics": None,
    "mid_1": "中单净额",
    "mid_net_last": "最近一日中单",
    "mid_pct": "中单净占比%",
    "missed_weekdays": None,
    "missing": "缺数字段",
    "mixed": "次要意图",
    "model_mix": None,
    "month": "月份",
    "month_end": "月末效应",
    "n": None,
    "n_holidays": "统计长假数",
    "n_titles": "分析资讯条数",
    "n_years": "统计年数",
    "natural_days": None,
    "near_cycle": "是否临近周期",
    "nearest_level_role": "档位角色",
    "nearest_resistance": "最近压力",
    "neg": None,
    "net_1": "最近一日净额",
    "net_10": "近10日净额",
    "net_20": "近20日净额",
    "net_5": "近5日净额",
    "net_5d": "近5日净额",
    "net_amount": "净流入额",
    "new_high": "创一年新高",
    "new_high_count": "创新高指数个数",
    "new_high_ratio": "创新高占比",
    "new_low": "创一年新低",
    "new_low_count": "创新低指数个数",
    "next_action": "操作提示",
    "next_report_period": "下一报告期",
    "next_windows": "未来时间窗",
    "north_tail": "北向持股",
    "note": None,
    "origin": "扇形原点",
    "outflow_days": "净流出天数",
    "p_big_down": "大跌概率",
    "p_big_up": "大涨概率",
    "p_down": "下跌概率",
    "p_flat": "横盘概率",
    "p_up": "上涨概率",
    "pageant": "形态",
    "pb": "PB",
    "pb_median_5y": "PB五年中位数",
    "pb_percentile": "PB历史分位%",
    "pcf": "PCF",
    "pct": None,
    "pct_1d": "近1日%",
    "pct_20d": "近20日%",
    "pct_5d": "近5日%",
    "pe_history_years": "估值统计年数",
    "pe_median_5y": "PE五年中位数",
    "pe_percentile": "PE历史分位%",
    "pe_static": "PE(静态)",
    "pe_ttm": "PE(TTM)",
    "pending": None,
    "percentile_method": None,
    "persistence": "持续性(日)",
    "phase": "所处阶段",
    "pmi": "制造业PMI",
    "policy": "产业政策",
    "pos": None,
    "pos_250d": "一年区间位置",
    "position": "位置",
    "position_in_day": "当日区间位置",
    "post10_ret": "节后10日收益%",
    "post5_ret": "节后5日收益%",
    "post_idx": None,
    "pre10_ret": "节前10日收益%",
    "pre5_ret": "节前5日收益%",
    "pre_idx": None,
    "prefix": None,
    "price": "最新价",
    "price_20d_chg": "近20日涨跌%",
    "price_5": None,
    "price_now": None,
    "price_stage": "价格阶段",
    "pricing_power": "定价权归属",
    "primary_trend": "主要趋势",
    "projected_bars": None,
    "ps": "PS",
    "quant_level": "量化活跃度",
    "r0_net": "超大单净额",
    "r2": "中单买入",
    "r250": None,
    "r3_net": "小单净额",
    "rebar": "螺纹钢",
    "recent_20d_pct": "近20日%",
    "recent_5d_pct": "近5日%",
    "regime": "市场状态",
    "related": None,
    "relative": "相对强度",
    "resistance_price": None,
    "resonance": "是否共振",
    "resonance_direction": "共振方向",
    "ret20": "近20日涨幅%",
    "ret5": "近5日涨幅%",
    "retail": "散户行为",
    "retail_score": "散户情绪分",
    "retracement": "回撤位",
    "retracements": "回撤位",
    "rows": None,
    "rs": None,
    "rsi": "RSI",
    "run": None,
    "same_day_corr": "同日相关",
    "search_hits": "搜索热度",
    "seasonal_date": "节气变盘日",
    "sector": "板块与题材",
    "sector_rank_pct": "板块强度分位",
    "segments": None,
    "series": None,
    "shibor_3m": "3个月Shibor%",
    "sigma": "波动率σ",
    "significant_bool": None,
    "slope": None,
    "small_1": "小单净额",
    "small_cap_pct": "中小盘近20日%",
    "small_net_last": "最近一日小单",
    "snap_pct": "当日涨跌%",
    "snapshot": None,
    "social_financing_yoy": "社融同比%",
    "source": None,
    "square_of_9": "江恩四方形",
    "stage": "所处阶段",
    "stage_detail": "阶段说明",
    "start": None,
    "state": None,
    "stock_20d": "个股近20日%",
    "stock_5d": "个股近5日%",
    "stock_60d": "个股近60日%",
    "strength_score": "趋势强度",
    "style": "市场风格",
    "style_spread": "大小盘差(pp)",
    "subsidy": "是否有补贴",
    "super_1": "超大单净额",
    "super_net_last": "最近一日超大单",
    "super_pct": "超大单净占比%",
    "super_pct_last": "超大单净占比%",
    "support_price": None,
    "swing_points": "摆动点",
    "table": None,
    "theme_names": "命中题材",
    "themes": "相关题材",
    "time": None,
    "time_cycles": "时间周期",
    "time_stage": "时间窗状态",
    "top10_ratio": "前十大占比%",
    "top10_tail": "前十大流通股东",
    "total": None,
    "total_shares": "总股本",
    "trade": "收盘价",
    "trend": "趋势",
    "trend_score": "趋势分",
    "turnover": "换手率%",
    "turnover_avg": "平均换手率%",
    "turnover_avg20_pct": "20日均换手%",
    "turnover_trend": "换手率趋势",
    "two_sessions": "两会窗口",
    "type": None,
    "up": None,
    "up_from_250d_low_pct": "距一年低点%",
    "upper_shadow": "上影线占比",
    "value": "数值",
    "value_fg": "指数值",
    "vol_percentile_inv": "波动率分位(反向)",
    "vol_ratio": "量比(5/20日)",
    "volatility_level": "波动率级别",
    "volume": "成交量",
    "volume_confirm": "量能验证",
    "volume_stage": "量能阶段",
    "votes": "各模型投票",
    "vpt": "量价时空",
    "vs_limit_up_pct": "距涨停%",
    "wave": "浪级",
    "weekday_best": "最强星期",
    "weekday_worst": "最弱星期",
    "weight": None,
    "win_rate": "胜率",
    "window": "窗口类型",
    "worst_day": "最弱星期",
    "worst_day_name": "最弱星期",
    "up_count": "上涨家数",
    "down_count": "下跌家数",
    "index_pct_avg": "指数平均涨幅%",
    "boards_hot": "涨停板块分布",
    "limit_up_count": "涨停家数",
    "max_continuous": "最高连板",
    "limit_up_date": "统计日期",
    "vix": "VIX恐慌指数",
    "vix_source": None,
    "as_of": "统计时间",
    "amp20_pct": "20日均振幅%",
    "annual_vol_pct": "年化波动率%",
    "anchor_price": "起算价",
    "anchor_type": "起算点类型",
    "ma20_slope_pct_per_day": "MA20斜率(%/日)",
    "drawdown_from_250d_high_pct": "距一年高点%",
    "trend_r2": "趋势拟合优度R²",
    "character": "个股性格",
    "top10_float_ratio": "前十大流通股东占比%",
    "trend_type": "趋势类型",
    "ma5": "MA5",
    "ma10": "MA10",
    "ma20": "MA20",
    "ma30": "MA30",
    "ma60": "MA60",
    "ma120": "MA120",
    "ma250": "MA250",
    "MA5": "MA5",
    "MA20": "MA20",
    "MA60": "MA60",
    "MA120": "MA120",
    "MA250": "MA250",
    "targets": "目标位",
    "extension": "扩展位",
    "time_windows": "时间窗口",
    "nearest_support": "最近支撑",
    "nearest_level": "最近档位",
    "position_pct": "区间位置",
    "fib_conformance": "波浪吻合度",
    "patterns": "量价组合",
    "lines": "角度线组",
    "sensitivity": "敏感性",
    "lh_ll": "高点降低+低点降低",
    "secondary_trend": "次级趋势",
    "bear_count": "看跌模型数",
    "chain": "链路",
    "flow_source": None,
    "turnover_pct_rank": "换手率分位",
    "guba_rank_change": "人气排名变化",
    "search_hits_change_pct": "搜索热度变化%",
    "is_hot_theme": "是否处于风口",
    "leader_stock": "板块龙头",
    "matched": "命中关键词",
    "cpi_yoy": "CPI同比%",
    "ppi_yoy": "PPI同比%",
    "lpr_1y": "LPR(1年)%",
    "lpr_5y": "LPR(5年)%",
    "m2_yoy": "M2同比%",
    "usd_cny": "美元人民币",
    "policy_score": "政策评分",
    "regulation_risk": "监管风险",
    "fallback_fields": "兜底字段",
    "ma_state": "均线状态",
    "index_pe_percentile": "指数估值分位",
    "drawdown_from_high_pct": "距高点回撤%",
    "index_close": "指数收盘",
    "open": "今开",
    "prev_close": "昨收",
    "pct_chg": "涨跌幅%",
    "amplitude": "振幅%",
    "volume_ratio": "量比",
    "name": "名称",
    "symbol": None,
    "fetched_at": None,
    "kind": None,
    "pe_series_note": "PE口径说明",
    "pb_series_note": "PB口径说明",
    "weekly": "周线",
    "monthly": "月线",
    "consensus": "理论综合",
    "score": "评分",
    "strength": "强度",
    "direction": "方向",
    "match": "是否共振",
    "low": "区间低点",
    "board": "板块",
    "boards": "所属板块",
    "count": "家数",
    "cycle": "周期",
    "ratio": "比率",
    "solar_terms": "二十四节气",
    "today": "当日",
    "weekday": "星期",
    "fib_window": "斐波那契时间窗",
    "in_window": "是否在时间窗内",
    "is_window": "是否处于窗口",
    "days_to": "距窗口天数",
    "days_to_deadline": "距披露截止日",
    "days_to_month_end": "距月末交易日",
    "risk": "风险提示",
    "hist_month_win_rate": "该月历史胜率",
    "prior_ret_pct": "经验先验收益%",
    "holder_num_change_pct": "股东户数变化%",
    "float_cap": "流通市值",
    "MA10": "MA10",
    "MA30": "MA30",
    "industry_name": "所属行业",
    "industry_pe_median": "行业中位PE",
    "industry_pb_median": "行业中位PB",
    "industry_peers_n": "同业公司数",
    "board_type": "板块类型",
    "keyword": "命中关键词",
    "keywords": "命中关键词",
    "matched_keywords": "命中关键词",
    "title": "标题",
    "solar_term": "节气",
    "term": "节气",
    "cycle_days": "周期(交易日)",
    "cycle_type": "周期类型",
    "is_near": "是否临近",
    "hits": "命中次数",
    "n_hits": "命中次数",
    "policy_hits": "命中政策",
    "matched_themes": "命中题材",
    "sigma_floor": None,
    "_dirs": None,
    "_retracement_pct": None,
    "_structure": None,
    "_bars": None,
    "_v1x1": None,
    "_atr": None,
    "_volume_ratio": None,
    "_ret5": None,
    "_price_position": None,
    "_streak": None,
    "_high": None,
    "_low": None,
    "_direction": None,
    "_avg_run": None,
    "_in_window": None,
    "_anchor_price": None,
    "_anchor_type": None,
}

#: 英文枚举值 → 中文（模块内部返回的英文标签）
_VALUE_LABELS = {
    "low": "低位", "high": "高位", "mid": "中位", "normal": "正常",
    "up": "上升", "down": "下降", "flat": "走平", "none": "无", "unknown": "未知",
    "bull": "看多", "bear": "看空", "range": "震荡", "weak": "趋势衰竭",
    "strong": "强", "medium": "中", "true": "是", "false": "否",
    "healthy": "健康", "degraded": "部分降级", "offline": "离线",
    "price_per_eps_approx": "按当前盈利回溯价格（近似法）",
    "eastmoney": "东方财富", "eastmoney_slist": "东方财富板块",
    "tencent": "腾讯财经", "sina": "新浪财经", "akshare": "AKShare",
    "none+eastmoney_ztpool+zbpool": "东财涨停/炸板池",
    "default": "配置默认值", "network": "网络接口",
    "synthetic": "合成数据（演示）", "kline_fallback": "日线兜底",
}


def _label(k: str) -> str | None:
    """字段名转中文；返回 None 表示该字段不展示。

    不展示的规则：
    * **下划线开头一律隐藏** —— 这类键是程序内部字段（``_holidays`` /
      ``_window`` / ``_years`` / ``_anchor`` / ``_dirs`` 等，模块用来传递
      中间结果），之前只隐藏了少数几个登记过的，导致 ``_holidays``、
      ``_window``、``_years`` 直接以英文出现在界面上；
    * 数据源标记（``*_source``）；
    * 显式登记为 None 的内部结构键（table/rows/items/新浪原始字段…）。
    """
    ks = str(k)
    if ks.startswith("_"):
        return None
    if ks in _FIELD_LABELS:
        return _FIELD_LABELS[ks]
    kl = ks.lower()
    if kl.endswith("_source") or kl == "source":
        return None
    return ks


def _df_zh(df: pd.DataFrame, *, drop_internal: bool = True) -> pd.DataFrame:
    """把 DataFrame 的英文列名批量译成中文，并丢弃内部技术列。

    模块内部统一用英文/下划线列名（便于程序处理），但展示给用户时必须是中文。
    这里在渲染前统一改名，避免每处表格单独维护。

    注意：源表**可能自带重复列名**（多个模块都可能输出 ``source`` 之类），
    所以先按位置挑选并展平重复列，再改名，最后确保结果没有重名列——
    否则 pandas 的 Arrow 后端会因重复列陷入递归。
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    picked: list = []          # [(列位置, 原列名)]
    seen: set = set()
    for i, c in enumerate(df.columns):
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if _label(key) is None and drop_internal:
            continue
        picked.append((i, key))
    if not picked:
        return df
    sub = df.iloc[:, [i for i, _ in picked]].copy()
    names, used = [], set()
    for _, key in picked:
        lab = _label(key)
        if lab is None:
            continue                    # 内部键不进表格
        base, n = lab, 2
        while lab in used:
            lab = "%s(%d)" % (base, n)
            n += 1
        used.add(lab)
        names.append(lab)
    sub.columns = names
    return sub


def _show_df(df: pd.DataFrame) -> None:
    """统一的数据表渲染：先中文化列名，再交给 st.dataframe。"""
    d = _df_zh(df)
    if d is None or (hasattr(d, "empty") and getattr(d, "empty", False)):
        st.caption("（本次未取得该项数据）")
        return
    st.dataframe(d, width="stretch", hide_index=True)


def _cat_factors(p: dict, category: str) -> list:
    """取某一类别的**全量**因子（优先用 factors_all，退回 forecast 里的 top8）。"""
    rows = [d for d in (p.get("factors_all") or []) if d.get("类别") == category]
    if rows:
        return sorted(rows, key=lambda d: -(d.get("强度", 0) * d.get("可信度", 0)))
    return [d for d in _all_factors(p) if d.get("类别") == category]


def _factor_block(p: dict, category: str, *, title: str = "") -> None:
    """统一渲染某一类别的全量因子表（八维明细各页共用）。"""
    rows = _cat_factors(p, category)
    if title:
        st.markdown("**%s**" % title)
    if not rows:
        st.caption("该类暂无因子输出。")
        return
    df = pd.DataFrame([{
        "因子": d.get("因子"), "方向": d.get("方向"),
        "强度": d.get("强度"), "可信度": d.get("可信度"),
        "观测值": str(d.get("观测值"))[:46],
        "解读": str(d.get("解读"))[:150],
        "理论依据": d.get("理论依据"),
    } for d in rows])
    _show_df(df)
    st.caption("共 %d 条因子（按强度×可信度排序）。" % len(rows))


def _clean_global_market(mk: dict) -> dict:
    """整理大盘/外围市场数据供前端展示（去掉冗余的K线对象）。"""
    if not isinstance(mk, dict):
        return {}
    snap = mk.get("snapshot") or {}
    rows = []
    for k, v in snap.items():
        if isinstance(v, dict):
            rows.append({
                "code": k, "name": v.get("name") or k,
                "group": v.get("group") or "",
                "price": v.get("price"), "pct_chg": v.get("pct_chg"),
                "last_date": v.get("last_date") or "",
            })
    lk = mk.get("linkage") or {}
    links = []
    for k, v in lk.items():
        if isinstance(v, dict):
            links.append({
                "code": k, "name": v.get("name") or k,
                "group": v.get("group") or "",
                "same_day_corr": v.get("same_day_corr"),
                "lead_corr": v.get("lead_corr"),
                "recent_5d_pct": v.get("recent_5d_pct"),
                "recent_20d_pct": v.get("recent_20d_pct"),
                "n": v.get("n"),
            })
    # 按领先相关性绝对值排序，最相关的排前面
    links.sort(key=lambda x: -abs(x.get("lead_corr") or 0))
    return {"snapshot": rows, "linkage": links,
            "relative": mk.get("relative") or {},
            "cross": mk.get("cross") or {},
            "detail": mk.get("detail") or ""}


def _render_value(v, *, depth: int = 0) -> None:
    """把任意嵌套值渲染成界面内容（**绝不静默丢弃**）。

    早期实现里 ``_kv_grid`` 会直接跳过 dict/list 值，导致"理论模型的摆动点/目标位/
    回撤位""恐慌贪婪分项""周期性窗口"等嵌套数据全部不显示，看起来像数据缺失。
    这里统一处理：标量 → 卡片；字典 → 递归卡片；字典列表 → 表格。
    """
    if depth > 6:
        # 极深结构兜底：也**不能**直接把 dict 转成字符串（会出现
        # "{'score': 0.692, 'turnover_avg': 0.19, ..." 这种原始字典字样），
        # 而是递归成"键：值"的形式。
        st.caption(_esc(_compact(v)))
        return
    if isinstance(v, dict):
        if not v:
            st.caption("（空）")
            return
        flat = {}
        for k2, v2 in v.items():
            lab = _label(k2)
            if lab is None:                 # 内部键（下划线开头等）不展示
                continue
            if isinstance(v2, (dict, list)) or v2 is None:
                continue
            if isinstance(v2, float) and v2 != v2:
                continue
            flat[lab] = v2
        if flat:
            _kv_grid(flat)
        for k2, v2 in v.items():
            if isinstance(v2, (dict, list)) and v2:
                lab = _label(k2)
                if lab is None:
                    continue
                st.markdown("**%s**" % _esc(lab))
                _render_value(v2, depth=depth + 1)
    elif isinstance(v, (list, tuple)):
        if not v:
            st.caption("（空）")
            return
        if isinstance(v[0], dict):
            _show_df(pd.DataFrame([_flatten_row(x) for x in v[:50]
                                   if isinstance(x, dict)]))
        else:
            vals = [x for x in v if x is not None
                    and not (isinstance(x, float) and x != x)]
            st.caption("、".join(_short(x) for x in vals[:40]) if vals else "（空）")
    else:
        if v is None or (isinstance(v, float) and v != v):
            st.caption("（本次未取得该项数据）")
        else:
            st.write(_short(v))


def _compact(v, depth: int = 0) -> str:
    """把任意嵌套值压成一行可读中文（**不使用 str(dict)**）。

    用于"表格单元格"与"极深结构兜底"这类只放得下一行文本的位置：
        {'score': 0.89, 'level': '高度控盘'}  →  "评分 0.89、控盘级别 高度控盘"
    """
    if depth > 3:
        return "…"
    if isinstance(v, dict):
        bits = []
        for k, x in v.items():
            lab = _label(k)
            if lab is None:
                continue
            bits.append("%s %s" % (lab, _compact(x, depth + 1)))
        return "、".join(bits)
    if isinstance(v, (list, tuple)):
        return "、".join(_compact(x, depth + 1) for x in v[:8])
    return _short(v)


def _flatten_row(d: dict) -> dict:
    """把一行里的嵌套值压成短字符串，便于放进表格（内部键不展示）。"""
    out = {}
    for k, v in d.items():
        lab = _label(k)
        if lab is None:
            continue
        if isinstance(v, (dict, list)):
            out[lab] = _compact(v)
        else:
            out[lab] = v
    return out


def _render_section(v: dict, *, depth: int = 0, label_map: dict | None = None) -> None:
    """渲染一个"分区"（保留旧名，内部统一走 :func:`render_fields`）。

    缺失的标量字段也会显示（灰底 + "—"），让用户看到"这项分析覆盖了、
    但本次数据源没提供"，而不是以为程序漏了。
    """
    render_fields(v, label_map=label_map, include_missing=True, depth=depth)


def _fmt_value(k, v) -> str:
    """按字段名自动选择数字格式。

    为什么要有它：以前每个页签都要手写 ``"PE(TTM)": _r(snap.get("pe_ttm"))``，
    既啰嗦又容易漏（漏了就把原始 float 显示成长串小数）。
    现在统一按"字段名后缀/关键词"决定格式：
        *%            → 百分数（0~1 的小数自动 ×100）
        _net/_amount  → 亿/万
        _shares       → 亿股/万股
        _ratio/_pct   → 小数
        其它数值      → %.4g
    """
    if isinstance(v, bool) or v is None:
        return _short(v)
    if isinstance(v, str):
        return _short(v)
    if isinstance(v, (dict, list, tuple)):
        return _compact(v)
    if not isinstance(v, (int, float)):
        return _short(v)
    f = float(v)
    if f != f:
        return "—"
    ks = str(k)
    if ks.endswith("%"):
        # 0~1 之间视为占比/分位（0.0182 → 1.8%），否则本身就是百分数
        return _pct01(f) if 0.0 <= f <= 1.0 else "%.2f%%" % f
    if re.search(r"(net|amount|money|flow|cap)", ks, re.I):
        return _big(f)
    if re.search(r"(shares|volume)", ks, re.I):
        return _big(f)
    if re.search(r"(ratio|slope|r2|brier|corr|beta|score|position|pos_|prob)", ks, re.I):
        return "%.3f" % f
    if abs(f) >= 1e8:
        return _big(f)
    if abs(f) >= 10000 and abs(f - round(f)) < 1e-9:
        return "{:,}".format(int(round(f)))
    return "%.4g" % f


def render_fields(v: dict, *, label_map: dict | None = None,
                  include_missing: bool = True, depth: int = 0) -> None:
    """**统一渲染入口**：把一个 dict 直接变成界面内容。

    这是收敛重构的核心。以前 9 个页签各自手写
    ``_kv_grid({中文名: _short(值)})``，导致三类问题反复出现：
        ① 嵌套字典被 ``_short()`` 变成原始字典字样并截断；
        ② 未登记的键以英文出现；
        ③ 长串小数（0.018166804293971925）。

    统一走这里之后，调用方**只传原始 dict**，由本函数负责：
        · 字段名 → 中文（label_map 优先，其次 _FIELD_LABELS）
        · 内部键（下划线开头 / 登记为 None）→ 隐藏
        · 数字格式 → _fmt_value 自动判断
        · 嵌套 dict → 递归卡片；dict 列表 → 表格
        · 缺失值 → 显示"—"（灰色虚框），让用户看清"分析了但没数据"
    """
    if not isinstance(v, dict) or not v:
        return
    if depth > 6:
        st.caption(_esc(_compact(v)))
        return
    lm = label_map or {}
    scalars: dict = {}
    nested: dict = {}
    for k, vv in v.items():
        lab = lm.get(k)
        if lab is None:
            lab = _label(k)
        if lab is None:
            continue                       # 内部键：隐藏
        if isinstance(vv, (dict, tuple)):
            if vv:
                nested[lab] = dict(vv)
        elif isinstance(vv, list):
            if vv:
                nested[lab] = vv
        else:
            scalars[lab] = vv
    if scalars:
        items = []
        for lab, vv in scalars.items():
            missing = (vv is None or (isinstance(vv, float) and vv != vv)
                       or (isinstance(vv, str) and not vv.strip()))
            if missing and not include_missing:
                continue
            items.append((lab, vv, missing))
        if items:
            _kv_grid_items(items)
    for lab, vv in nested.items():
        st.markdown("**%s**" % _esc(lab))
        if isinstance(vv, list):
            _render_value(vv, depth=depth + 1)
        else:
            render_fields(vv, label_map=lm, include_missing=include_missing,
                          depth=depth + 1)


def _kv_grid_items(items: list) -> None:
    """底层渲染：把 ``[(标签, 值, 是否缺失)]`` 画成卡片网格。"""
    html = ["<div class='bigprobs'>"]
    for k, v, missing in items:
        cls = "pcard miss" if missing else "pcard"
        val = "—" if missing else _esc(_fmt_value(k, v))
        html.append("<div class='%s'><div class='lbl'>%s</div>"
                    "<div class='val' style='font-size:15px'>%s</div></div>"
                    % (cls, _esc(k), val))
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _clean_learning_report(symbol: str) -> dict:
    """取该股票的自我学习报告（已核账样本、命中率、校准系数）。"""
    try:
        from core import learning as L

        rep = L.calibration_report(symbol) or {}
        return {
            "detail": rep.get("detail") or "",
            "last_update": rep.get("last_update") or "",
            "evaluated": rep.get("evaluated", 0),
            "pending": rep.get("pending", 0),
            "table": [r for r in (rep.get("table") or []) if isinstance(r, dict)],
            "dimension_weights": rep.get("dimension_weights") or {},
        }
    except Exception as exc:  # noqa: BLE001
        log.debug("读取学习报告失败: %s", exc)
        return {}


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
    """递归清理成可缓存/可 JSON 化的结构。

    **注意**：早期版本这里用很浅的 depth（1~2），导致"情绪/宏观/理论/季节性"里
    的嵌套字典（如恐慌贪婪分项、理论模型的摆动点/目标位/回撤位、周期性窗口）
    在传给前端之前就被整片抹成 None——用户看到大片空框格，而原始数据其实是有值的。
    现在改为保留全部嵌套层级（只做深度上限保护，防止异常深的结构拖慢渲染），
    并对 DataFrame 等不可序列化对象单独降级。
    """
    if depth <= 0:
        return None if isinstance(d, (dict, list)) else _jsonable(d)
    if isinstance(d, dict):
        return {k: _deep_clean(v, depth - 1) for k, v in d.items()
                if not isinstance(v, pd.DataFrame)}
    if isinstance(d, (list, tuple)):
        return [_deep_clean(x, depth - 1) for x in d]
    return _jsonable(d)


def _jsonable(v):
    """把值转成可安全展示/缓存的形式（不丢信息）。"""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, pd.DataFrame):
        return None
    if isinstance(v, (list, tuple, dict)):
        try:
            return _deep_clean(v, depth=8)
        except Exception:  # noqa: BLE001
            return str(v)[:200]
    try:                                  # numpy / Timestamp 等
        if hasattr(v, "item"):
            return v.item()
        if hasattr(v, "isoformat"):
            return v.isoformat()
    except Exception:  # noqa: BLE001
        pass
    return str(v)[:200]


def _frame_tail(df, n: int):
    """把财务表转成前端可用的 dict，并保证关键列一定被带上。

    之前的实现只挑了固定几列，遇到财务模块输出的**中文列名变体**就会整列丢失，
    导致界面显示"未取得财务数据"。这里改为：优先用规范列，同时兼容别名，
    并额外附带估值分位等派生字段。
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    # 规范列名 → 可能的别名（按优先级）
    alias = {
        "报告期": ("报告期", "报告日期", "REPORT_DATE", "报告期类型"),
        "营业收入": ("营业收入", "营业总收入", "营收"),
        "营收同比": ("营收同比", "营业收入同比", "营收同比增长", "营收同比(%)"),
        "净利润": ("净利润", "归母净利润", "归属净利润"),
        "净利同比": ("净利同比", "净利润同比", "净利同比增长", "净利润同比(%)"),
        "单季营收": ("单季营收", "单季度营业收入"),
        "单季净利": ("单季净利", "单季净利润", "单季度净利润"),
        "营收环比": ("营收环比", "营业收入环比", "单季营收环比"),
        "净利环比": ("净利环比", "净利润环比", "单季净利环比"),
        "ROE": ("ROE", "净资产收益率", "摊薄ROE", "加权ROE"),
        "毛利率": ("毛利率", "销售毛利率"),
        "净利率": ("净利率", "销售净利率"),
        "资产负债率": ("资产负债率",),
        "经营现金流": ("经营现金流", "经营活动现金流", "经营活动产生的现金流量净额"),
        "每股收益": ("每股收益", "EPS", "基本每股收益"),
        "每股净资产": ("每股净资产", "BPS"),
    }
    cols = list(df.columns)

    def pick(names):
        for nm in names:
            if nm in cols:
                return nm
        # 宽松匹配：包含关系
        for nm in names:
            for c in cols:
                if nm in str(c):
                    return c
        return None

    out_cols, rename = [], {}
    for std, names in alias.items():
        hit = pick(names)
        if hit is not None:
            out_cols.append(hit)
            rename[hit] = std
    if not out_cols:
        # 实在匹配不上，就把所有数值列都带上（保证不空手而归）
        out_cols = cols[:14]
        rename = {c: c for c in out_cols}
    sub = df[out_cols].tail(n).rename(columns=rename)
    # 展开报告期，避免 Timestamp 在界面上显示成一长串
    if "报告期" in sub.columns:
        sub["报告期"] = sub["报告期"].astype(str).str.slice(0, 10)
    return sub.to_dict("list")


def _capital_brief(cap: dict) -> dict:
    out: dict = {}
    flow = cap.get("flow")
    if isinstance(flow, pd.DataFrame) and not flow.empty:
        try:
            if "main_net" in flow.columns:
                out["main_net_5d"] = float(flow["main_net"].tail(5).sum())
                out["main_net_10d"] = float(flow["main_net"].tail(10).sum())
                out["main_net_20d"] = float(flow["main_net"].tail(20).sum())
                out["main_net_last"] = float(flow["main_net"].iloc[-1])
            for col, key in (("super_net", "super_net_last"), ("big_net", "big_net_last"),
                             ("mid_net", "mid_net_last"), ("small_net", "small_net_last"),
                             ("main_pct", "main_pct_last"),
                             ("super_pct", "super_pct_last")):
                if col in flow.columns:
                    try:
                        out[key] = float(pd.to_numeric(flow[col], errors="coerce").iloc[-1])
                    except Exception:  # noqa: BLE001
                        pass
            if "date" in flow.columns:
                out["flow_last_date"] = str(flow["date"].iloc[-1])[:10]
            out["flow_source"] = str(flow.attrs.get("source") or "") or None
            out["flow_rows"] = int(len(flow))
        except Exception as exc:  # noqa: BLE001
            log.debug("主力资金摘要失败: %s", exc)
    for key, src in (("main_force_profile", cap.get("main_force_profile") or cap.get("profile")),
                     ("control_degree", (cap.get("control_degree") or {})
                      if isinstance(cap.get("control_degree"), dict) else None),
                     ("minute_flow", (cap.get("minute_flow") or {})
                      if isinstance(cap.get("minute_flow"), dict) and cap.get("minute_flow")
                      else None)):
        if isinstance(src, dict) and src:
            out[key] = _clean_dict(src, depth=1)
    for key, df in (("holders_tail", cap.get("holders")), ("margin_tail", cap.get("margin")),
                    ("lhb", cap.get("lhb")), ("north_tail", cap.get("north")),
                    ("top10_tail", cap.get("top10"))):
        if isinstance(df, pd.DataFrame) and not df.empty:
            try:
                out[key] = df.tail(5).astype(object).where(pd.notna(df.tail(5)), None) \
                    .to_dict("list")
            except Exception:  # noqa: BLE001
                out[key] = df.tail(5).to_dict("list")
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


def _apply_symbol(code: str) -> None:
    """按钮/单选框的回调：把选中的代码写入输入框并触发分析。

    必须在 **回调** 里写 ``st.session_state["symbol_input"]``——
    Streamlit 不允许在部件实例化之后修改它的 session_state 键，
    这也是之前手机端"代码解析失败：
    st.session_state.symbol_input cannot be modified after the widget ..." 的原因。
    """
    st.session_state["symbol_input"] = code
    st.session_state["run_analysis"] = True
    st.session_state.pop("name_pick", None)


def sidebar() -> dict:
    st.sidebar.title("个股涨跌概率预测")
    st.sidebar.caption("core v%s ｜ 8 个周期 ｜ 8 大维度因子" % CORE_VERSION)

    st.session_state.setdefault("symbol_input", "600519")

    symbol = st.sidebar.text_input("股票代码 / 名称", key="symbol_input",
                                   help="支持 600519、sh600519、000001.SZ、00700.HK、AAPL，"
                                        "也可直接输入中文名或拼音（如 贵州茅台 / maotai）")
    if st.sidebar.button("开始分析", type="primary", width="stretch"):
        st.session_state["run_analysis"] = True
        cached_analysis.clear()

    with st.sidebar.expander("常用股票", expanded=False):
        cols = st.columns(2)
        for i, item in enumerate(POPULAR):
            code = item.split()[0]
            cols[i % 2].button(code, key="pop_%s" % code, width="stretch",
                               on_click=_apply_symbol, args=(code,))

    st.sidebar.markdown("---")
    with st.sidebar.expander("分析设置", expanded=False):
        use_ml = st.checkbox(
            "启用 AI 模型", value=True,
            help="用该股自身历史数据训练三个内置模型（逻辑回归 / 梯度提升树 / "
                 "历史相似形态类比），做扩展窗口前向验证后与规则模型融合。"
                 "纯本地计算，不需要任何 API Key、账号或外部服务。")
        st.caption("ℹ️ AI 模型**在本机/服务器上自己训练**，无需填写 API 密码，"
                   "也不联网调用任何第三方 AI 服务。开启后每只股票多花约 10~30 秒。")
        demo = st.checkbox("演示模式（用合成数据，不联网）", value=False,
                           help="无网络或数据源不可用时，用合成数据验证流程与界面")
        days = st.slider("历史数据天数", 400, 3200, 1600, step=200,
                         help="回测与 AI 训练都依赖历史长度，长周期预测建议 ≥1600")
        self_learning = st.checkbox(
            "启用自我学习", value=True,
            help="每次预测写入档案，周期走完后用真实价格核账，"
                 "并据此校准概率与置信度。样本不足时自动用历史数据回填。关闭后不记录、不校准。")
        do_backtest = st.checkbox("执行历史回测（校准置信度）", value=True,
                                  help="用历史数据检验规则分与未来收益的相关性，"
                                       "并据此对每个周期的置信度打分")
    st.sidebar.markdown("---")
    if st.sidebar.button("清空数据缓存", width="stretch"):
        n = cache_clear()
        cached_analysis.clear()
        st.sidebar.success("已清理 %d 个缓存文件" % n)
    with st.sidebar.expander("关于 / 方法说明"):
        st.markdown("""
**预测流程**

1. 取数（日/周/月线、实时快照、大盘、资金流、财务、新闻…）
2. 八大维度分析，每个模块只输出统一的"因子"
   （方向 / 强度 / 可信度 / 理论依据）
3. 按 **周期权重矩阵** 汇总因子：短线重资金与技术，
   长线重基本面与宏观
4. 方向分 → logit 概率；波动率 × √t → 幅度区间
5. 内置 AI 模型按周期融合（**无需 API Key**）
6. 多周期一致性检查 + 历史回测校准置信度

**关于 AI**

三个模型全部用 numpy 自己实现，在本地训练：
逻辑回归、直方图梯度提升树、历史相似形态类比（KNN）。
**不调用任何外部 AI 服务，不需要填写密钥。**

**重要**：模型输出的是**概率分布**，不是确定答案。
当证据不足时它会给出接近 50% 的概率并降低置信度——
这是"看不清楚"的诚实表达，而不是故障。
""")
    return {"symbol": symbol, "use_ml": use_ml, "demo": demo, "days": days,
            "backtest": do_backtest, "self_learning": self_learning}


# --------------------------------------------------------------------------- #
# 渲染组件
# --------------------------------------------------------------------------- #
def pcard(label: str, value: str, sub: str = "", cls: str = "") -> str:
    return ("<div class='pcard'><div class='lbl'>%s</div>"
            "<div class='val %s'>%s</div><div class='sub'>%s</div></div>"
            % (label, cls, value, sub))


def render_header(p: dict) -> None:
    """顶部标题区：股票名称 + 代码 + 数据概要 + 状态标签。"""
    import html as _h

    name = (p.get("name") or "").strip()
    code = (p.get("symbol") or "").strip()
    # 名称缺失或与代码相同时，只显示代码，避免出现 "（）" 这种空括号
    title = _h.escape(name) if name and name != code else _h.escape(code or "未知标的")
    code_html = ("<span class='code'>%s</span>" % _h.escape(code)) \
        if (name and name != code and code) else ""

    tags = []
    if p.get("is_demo"):
        tags.append("<span class='tag warn'>注意：演示模式（合成数据，不可用于实盘）</span>")
    snap = p.get("snapshot") or {}
    if snap.get("source"):
        tags.append("<span class='tag'>数据源 %s</span>" % _h.escape(str(snap["source"])))
    srcs = p.get("meta_sources") or {}
    if srcs:
        tags.append("<span class='tag'>%d 个分析模块</span>" % len(srcs))
    tags.append("<span class='tag'>%d 条因子</span>" % p.get("n_factors", 0))
    tags.append("<span class='tag'>耗时 %.0fs</span>" % p.get("elapsed", 0))
    if (p.get("meta_errors") or []):
        tags.append("<span class='tag warn'>%d 个模块未完成</span>"
                    % len(p["meta_errors"]))

    st.markdown(
        "<div class='hero stock'>"
        "<h1>%s%s</h1>"
        "<div class='meta'>数据截止 <b>%s</b>　·　现价 <b>%s</b>　·　"
        "预测周期 1天 / 5天 / 10天 / 20天 / 1个月 / 1季度 / 半年 / 1年</div>"
        "<div class='tags'>%s</div>"
        "</div>" % (
            title, code_html,
            _h.escape(str(p.get("as_of") or "—")),
            ("%.2f" % p["price"]) if p.get("price") else "—",
            "".join(tags)),
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
    ai_used = 0.0
    for f in p.get("forecasts", []):
        cls = "up" if f["p_up"] >= 0.5 else "down"
        ai_used = max(ai_used, f["model_mix"].get("AI模型", 0))
        cards.append(pcard(
            f["label"], "%.1f%%" % (f["p_up"] * 100), cls=cls,
            sub="%s ｜ %+.2f%%<br>区间 %+.1f%%~%+.1f%% ｜ 置信 %.0f%%" % (
                f["trend"], f["exp_return"], f["ci_low"], f["ci_high"],
                f["confidence"] * 100)))
    st.markdown("<div class='bigprobs'>%s</div>" % "".join(cards), unsafe_allow_html=True)

    # 明确回答"AI 是否需要配置密钥"这个问题
    if ai_used > 0:
        st.markdown(
            "<div class='tip'><b>AI 模型已参与本次预测（最高权重占比 %.0f%%）</b>。"
            "它是在本机用该股历史数据<b>现场训练</b>的三个内置模型"
            "（逻辑回归 / 梯度提升树 / 历史相似形态类比），"
            "<b>不需要 API Key、账号或任何外部 AI 服务</b>。"
            "想关掉它可在左侧「分析设置」里取消勾选「启用 AI 模型」，速度会快一些。</div>"
            % (ai_used * 100), unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='tip'>本次预测<b>未使用 AI 模型</b>（可能被关闭，或历史样本不足）。"
             "AI 模型是本程序内置的，<b>无需 API Key</b>，"
             "在左侧「分析设置」勾选「启用 AI 模型」即可参与预测。</div>",
            unsafe_allow_html=True)


def _f(p: dict, key: str):
    for f in p.get("forecasts", []):
        if f["horizon"] == key:
            return f
    return None


def render_forecast_table(p: dict) -> None:
    """八周期预测表。

    宽屏显示完整列；窄屏（手机/小窗口）自动只显示核心列，
    其余列放到下面的"完整数据"折叠区，避免表格被右侧裁切。
    """
    full_rows, core_rows = [], []
    for f in p.get("forecasts", []):
        full_rows.append({
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
        core_rows.append({
            "周期": f["label"], "结论": f["trend"],
            "上涨概率": "%.1f%%" % (f["p_up"] * 100),
            "期望涨跌幅": "%+.2f%%" % f["exp_return"],
            "波动区间": "%+.1f%%~%+.1f%%" % (f["ci_low"], f["ci_high"]),
            "置信度": "%.0f%%" % (f["confidence"] * 100),
        })
    # 用容器查询不支持，改用列数判断：≥12 列时优先给完整表（可横向滚动），
    # 同时始终提供核心表，手机用户默认看到核心表。
    st.markdown("**核心结论**")
    _show_df(pd.DataFrame(core_rows))
    with st.expander("查看完整数据（区间 / 大涨大跌概率 / 波动率 / AI参与度）",
                     expanded=False):
        _show_df(pd.DataFrame(full_rows))
    st.caption("说明：**大涨/大跌概率**按各周期阈值计算（1天 ±3%、20天 ±12%、1年 ±50%）；"
               "**置信度**已根据历史回测结果自动折扣。表格可左右滑动查看全部列。")


#: 每张图**各自**的说明（早期版本六张图共用同一句说明，等于没说明）
_CHART_META = {
    "price": (
        "K线与均线",
        "最近约 180 个交易日的蜡烛图（红涨绿跌）。叠加 MA5/MA20/MA60 三条均线与布林带，"
        "横向虚线为算法识别出的支撑位（蓝）与压力位（橙）。"
        "看点在均线排列方向、价格与布林带上下轨的位置关系，以及离支撑/压力还有多少空间。",
    ),
    "indicators": (
        "技术指标",
        "自上而下依次是 MACD（含红绿柱与 DIF/DEA）、KDJ（含 80/20 超买超卖线）、"
        "RSI6 与 RSI14（含 70/30 线）。用于判断动能强弱与超买超卖，"
        "重点看柱状能量变化与指标金叉/死叉的位置。",
    ),
    "multi_horizon": (
        "周期对比",
        "左图：8 个预测周期的上涨概率（虚线为 50% 基准，红=偏多、绿=偏空）；"
        "右图：对应的期望涨跌幅。两栏逐行对应同一周期，"
        "用于一眼看出短中长期结论是否一致（共振）还是互相矛盾。",
    ),
    "gauge": (
        "概率仪表",
        "把各周期上涨概率画成横向仪表条：红色越满表示看涨把握越大，绿色越满表示越看跌。"
        "注意右端数字只是概率，不等于幅度——幅度请看「周期对比」或「置信锥」。",
    ),
    "cone": (
        "置信锥",
        "横轴为 8 个周期（等距排列，非按天数比例），纵轴为预期涨跌幅。"
        "深色折线是期望涨跌幅，蓝色阴影是约 72% 概率的区间。"
        "**周期越长阴影越宽**，直观体现「时间越长越不可预测」。",
    ),
    "contribution": (
        "因子贡献",
        "未来 20 天预测中各因子对方向分的贡献度（%）：红条为看涨贡献、绿条为看跌贡献，"
        "按贡献绝对值排序。这张图回答的是「**这个结论主要是被什么推出来的**」，"
        "用于判断结论是否建立在可靠证据上。",
    ),
}


def render_charts(p: dict) -> None:
    ch = p.get("charts") or {}
    if not ch:
        st.info("暂无图表（数据不足）。")
        return
    tabs = st.tabs([_CHART_META[k][0] for k in _CHART_META if k in ch])
    i = 0
    for key, (label, desc) in _CHART_META.items():
        if key not in ch:
            continue
        with tabs[i]:
            st.image(ch[key], width="stretch")
            st.markdown("<div class='tip'>%s</div>" % _esc(desc), unsafe_allow_html=True)
        i += 1


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
        st.markdown("**正向贡献（看涨）**")
        for d in f.get("drivers", [])[:8]:
            st.markdown("<span class='pill bull'>%s</span> **%s** ｜ %s "
                        "<span style='color:#888;font-size:11.5px'>[%s] 强度%.2f 可信度%.2f"
                        " 贡献%+.1f%%</span>" % (
                            _esc(d.get("类别", "")), _esc(d.get("因子", "")),
                            _esc(d.get("解读", "")), _esc(d.get("理论依据", "")),
                            float(d.get("强度", 0)), float(d.get("可信度", 0)),
                            float(d.get("贡献", 0))), unsafe_allow_html=True)
    with c2:
        st.markdown("**反向贡献（看跌/风险）**")
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
            _show_df(pd.DataFrame(rows))


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
        # 行情快照：直接传原始字段，标签与数字格式由 render_fields 统一处理
        render_fields(snap, label_map={"name": "股票名称", "code": "股票代码",
                                        "trade": "现价", "price": "现价"})
        if stage:
            st.markdown("**所处阶段与特性判定**")
            stage_label = {
                "stage": "所处阶段", "stage_detail": "阶段说明", "detail": "阶段说明",
                "trend_type": "趋势类型", "trend_r2": "趋势拟合优度R²",
                "volatility_level": "波动率级别", "annual_vol_pct": "年化波动率%",
                "pricing_power": "定价权归属", "top10_float_ratio": "前十大流通股东占比%",
                "quant_level": "量化活跃度", "character": "个股性格",
                "pos_250d": "一年区间位置", "drawdown_from_250d_high_pct": "距一年高点%",
                "up_from_250d_low_pct": "距一年低点%",
                "ma20_slope_pct_per_day": "MA20斜率(%/日)",
                "turnover_avg20_pct": "20日均换手%", "amp20_pct": "20日均振幅%",
                "code": "代码", "name": "名称", "price": "价格",
                "control_degree": "控盘程度", "main_force_profile": "主力行为特征",
            }
            # 用 _render_section 而不是把自己手工拼字典：
            # 旧写法 `{stage_label.get(k, k): _short(v) for ...}` 有两个毛病
            #   ① 嵌套字典（如 control_degree 控盘程度）被 _short() 转成
            #      "{'score': 0.89, 'turnover_avg': 0.19, 'turnover_trend': nan…"
            #      这种原始字典字样，还被截断；
            #   ② 未登记的键直接以英文出现。
            _render_section(stage, label_map=stage_label)
        if vol:
            st.markdown("**波动率结构**")
            render_fields(vol, label_map={"daily_pct": "日波动率%",
                                           "annual_pct": "年化波动率%",
                                           "vol_20": "20日波动%", "vol_60": "60日波动%",
                                           "vol_ratio": "短长波动比", "regime": "波动状态",
                                           "atr_pct": "ATR%"})
        _factor_block(p, "个股特性/阶段", title="该维度生成的因子")

    with tab2:
        st.markdown("**技术面因子全清单**（含 KDJ / MACD / RSI / BOLL / SAR / DMI / 筹码 / 形态等）")
        _factor_block(p, "技术面")
        st.caption("说明：因子只在**有明确信号时**才生成。若某指标（如 KDJ）没出现在列表里，"
                   "说明它当前既没有金叉/死叉，也没进入超买超卖区，属正常情况。")

    with tab3:
        if val:
            render_fields(val, label_map={"pe_static": "PE(静态)",
                                           "pe_median_5y": "PE五年中位数",
                                           "pb_median_5y": "PB五年中位数",
                                           "eps_ttm": "每股收益TTM", "bps": "每股净资产"})
            if val.get("percentile_method"):
                st.caption("估值分位口径：%s（%s）" % (
                    val["percentile_method"], val.get("pe_series_note", "")))
        else:
            st.warning("未取得估值数据（PE/PB）。可能是实时快照接口不可用，"
                       "可稍后重试或点击左侧「清空数据缓存」。")
        fin = p.get("financials_tail")
        if fin and any(len(v) for v in fin.values()):
            st.markdown("**财务摘要（最近若干期）**")
            _show_df(pd.DataFrame(fin))
            st.caption("报告期口径：A 股财报为**年内累计**，"
                       "单季/环比列已由财务模块拆分计算。")
        else:
            st.warning("未取得财务数据。财务摘要依赖 akshare，"
                       "若该库未安装或接口受限会显示为空；"
                       "此时基本面因子会退化为低可信度。")
        _factor_block(p, "基本面", title="该维度生成的因子")

    with tab4:
        # ① 主力操盘意图（本次新增维度，放在最前面）
        mi = p.get("main_intent") or {}
        if mi:
            st.markdown("**主力当前操盘意图**")
            d = mi.get("direction", 0)
            cls = "up" if d > 0 else ("down" if d < 0 else "flat")
            st.markdown(
                "<div class='tip'><b>判定：<span class='%s'>%s</span></b>"
                "　意图分 <b>%+.2f</b>　强度 %.2f　可信度 %.0f%%"
                "　持续性 %s 日%s</div>" % (
                    cls, _esc(mi.get("intent_cn") or "—"), float(mi.get("score") or 0),
                    float(mi.get("strength") or 0), float(mi.get("confidence") or 0) * 100,
                    mi.get("persistence") or 0,
                    ("　次要意图：" + _esc(mi.get("mixed"))) if mi.get("mixed") else ""),
                unsafe_allow_html=True)
            if mi.get("next_action"):
                st.info("操作提示：%s" % mi["next_action"])
            ev = mi.get("evidence") or []
            if ev:
                st.markdown("**判定依据**")
                for e in ev[:6]:
                    st.markdown("- %s" % _esc(e))
            st.caption("原理：单看主力净流入没有意义——同样净流入，股价温和上涨是「吃货」，"
                       "大涨可能是「拉高出货」前奏，下跌反而是「洗盘」。"
                       "本判定综合了资金方向、价格表现、量能变化与股价位置。")
            ser = p.get("intent_series") or []
            if ser:
                with st.expander("查看最近 %d 天的意图变化" % len(ser)):
                    _show_df(pd.DataFrame([{"日期": x.get("date"),
                                            "意图": x.get("intent_cn"),
                                            "意图分": x.get("score")} for x in ser]))
        render_fields({k: cap.get(k) for k in
                        ("main_net_5d", "main_net_10d", "main_net_20d",
                         "main_net_last", "super_net_last", "big_net_last",
                         "main_pct_last", "super_pct_last",
                         "flow_last_date", "flow_rows")},
                       label_map={"main_net_5d": "近5日主力净额",
                                  "main_net_10d": "近10日主力净额",
                                  "main_net_20d": "近20日主力净额",
                                  "main_net_last": "最近一日主力净额",
                                  "super_net_last": "最近一日超大单",
                                  "big_net_last": "最近一日大单",
                                  "main_pct_last": "主力净占比%",
                                  "super_pct_last": "超大单净占比%"})
        if not any(cap.get(k) is not None for k in
                   ("main_net_5d", "main_net_20d", "main_net_last")):
            st.warning("主力资金数据为空。资金流依赖东财接口（本机网络常被重置），"
                       "程序会自动降级到新浪历史资金流；若两者都失败则为空，"
                       "此时资金面因子会退化为低可信度。")
        prof = cap.get("main_force_profile") or {}
        if prof:
            st.markdown("**主力行为判定**")
            render_fields(prof)
        cd = cap.get("control_degree") or {}
        if cd:
            st.markdown("**控盘程度**")
            render_fields(cd)
        for key, label in (("holders_tail", "股东户数（户均持股）"),
                           ("margin_tail", "融资融券"), ("lhb", "龙虎榜"),
                           ("north_tail", "北向持股")):
            v = cap.get(key)
            if v:
                st.markdown("**%s**" % label)
                _show_df(pd.DataFrame(v))
            else:
                st.caption("（%s：本次未取到数据）" % label)
        _factor_block(p, "资金面/主力", title="该维度生成的因子")

    with tab5:
        if sent:
            label_map = {
                "market": "全市场情绪",
                "sector": "所属板块与题材热度",
                "retail": "散户行为（反向指标）",
                "fear_greed": "恐慌/贪婪指数",
            }
            for k, v in sent.items():
                if isinstance(v, dict) and v:
                    _lab = label_map.get(k)
                    if _lab is None:
                        _lab = _label(k)
                    if _lab is None:
                        continue
                    st.markdown("**%s**" % _esc(_lab))
                    if k == "market":
                        st.caption("以下为**全市场**统计（大盘情绪），不是该股自身的指标；"
                                   "其中「涨停板块分布」用于判断当日资金在炒什么方向。")
                    if k == "retail":
                        st.caption("散户行为是**反向指标**：人气/搜索骤升往往对应短线见顶。")
                    _render_section(v)
                elif isinstance(v, (list, tuple)) and v:
                    _lab2 = label_map.get(k)
                    if _lab2 is None:
                        _lab2 = _label(k)
                    if _lab2 is None:
                        continue
                    st.markdown("**%s**" % _esc(_lab2))
                    _render_value(v)
                elif isinstance(v, (str, int, float, bool)) and v is not None:
                    st.write("%s：%s" % (_label(k), _fmt_value(k, v)))
        else:
            st.info("未取得情绪数据。")
        boards = p.get("sector_boards") or []
        if boards:
            st.markdown("**该股所属板块（均与个股直接相关）**")
            _show_df(pd.DataFrame(boards))
        _factor_block(p, "情绪/博弈", title="该维度生成的因子")

    with tab6:
        if mac:
            mac_label = {"macro": "宏观经济指标", "policy": "产业政策",
                         "commodity": "上下游商品价格（行业相关）",
                         "commodity_relevant": "商品相关性筛选说明",
                         "market_regime": "大盘水位", "industry_names": "所属行业",
                         "industry": "行业周期", "notes": "数据说明"}
            # ① 大盘与外围股市（本次新增维度）
            gm = p.get("market_global") or {}
            # 商品（按行业筛选后只展示相关品种，避免出现与该股无关的原油/铜）
            rel = mac.get("commodity_relevant") or {}
            rel_items = rel.get("items") or {}
            if gm.get("snapshot"):
                st.markdown("**大盘与外围股市（A股 / 港股 / 美股 / 风险偏好）**")
                if gm.get("detail"):
                    st.caption(gm["detail"])
                grp_label = {"a_share": "A股大盘", "hk": "港股", "us": "美股（隔夜）",
                             "risk": "风险偏好", "a50": "A50期货"}
                _show_df(pd.DataFrame([{
                    "分类": grp_label.get(r.get("group"), r.get("group") or "—"),
                    "指数": r.get("name"), "最新价": _r(r.get("price")),
                    "涨跌幅%": _r(r.get("pct_chg")), "数据日期": str(r.get("last_date") or "")[:10],
                } for r in gm["snapshot"]]))
                rel = gm.get("relative") or {}
                if rel:
                    render_fields(rel, label_map={"beta_250d": "Beta(250日)",
                                                   "corr_250d": "相关性(250日)"})
                    if rel.get("corr_note"):
                        st.caption("提示：%s" % rel["corr_note"])
                # ---- A股指数之间的相互印证 / 顶底背离 ----
                cx = gm.get("cross") or {}
                if cx.get("agreement"):
                    st.markdown("**A股各指数相互印证与背离**")
                    if cx.get("detail"):
                        st.caption(cx["detail"])
                    _show_df(pd.DataFrame([{
                        "指数": r.get("name"),
                        "近1日%": _r(r.get("pct_1d")),
                        "近5日%": _r(r.get("pct_5d")),
                        "近20日%": _r(r.get("pct_20d")),
                        "创一年新高": "是" if r.get("new_high") else "否",
                        "创一年新低": "是" if r.get("new_low") else "否",
                    } for r in (cx.get("rows") or [])]))
                    ag = cx.get("agreement") or {}
                    _kv_grid_items([
                        ("1日一致度", "%d涨%d跌（%.0f%%）" % (
                            (ag.get("1日") or {}).get("up", 0),
                            (ag.get("1日") or {}).get("down", 0),
                            ((ag.get("1日") or {}).get("ratio") or 0) * 100), False),
                        ("5日一致度", "%d涨%d跌（%.0f%%）" % (
                            (ag.get("5日") or {}).get("up", 0),
                            (ag.get("5日") or {}).get("down", 0),
                            ((ag.get("5日") or {}).get("ratio") or 0) * 100), False),
                        ("20日一致度", "%d涨%d跌（%.0f%%）" % (
                            (ag.get("20日") or {}).get("up", 0),
                            (ag.get("20日") or {}).get("down", 0),
                            ((ag.get("20日") or {}).get("ratio") or 0) * 100), False),
                        ("背离判定", {"top": "顶背离（上涨缺乏广泛性）",
                                      "bottom": "底背离（下跌过度）",
                                      "none": "无背离"}.get(cx.get("divergence"), "—"),
                         False),
                        ("市场风格", cx.get("style") or "—", False),
                        ("大盘权重近20日%", cx.get("large_cap_pct"), False),
                        ("中小盘近20日%", cx.get("small_cap_pct"), False),
                    ])
                    st.caption("依据**道氏理论**：主要趋势必须得到各指数相互印证。"
                               "若只有个别指数创新高/新低而其余未跟随，即为背离，"
                               "提示上涨不牢或下跌过度。")
                if gm.get("linkage"):
                    st.markdown("**联动关系（用于判断该重点参考哪个指数）**")
                    st.caption("「同日相关」= 指数当日涨跌与该股当日涨跌的相关性；"
                               "「领先相关」= 指数**前一日**涨跌与该股当日涨跌的相关性，"
                               "**领先相关才是可用于预测的关系**（美股隔夜效应即属此类）。")
                    _show_df(pd.DataFrame([{
                        "分类": grp_label.get(x.get("group"), x.get("group") or "—"),
                        "指数": x.get("name"),
                        "同日相关": _r(x.get("same_day_corr"), 3),
                        "领先相关": _r(x.get("lead_corr"), 3),
                        "近5日%": _r(x.get("recent_5d_pct")),
                        "近20日%": _r(x.get("recent_20d_pct")),
                        "样本天数": x.get("n"),
                    } for x in gm["linkage"]]))
            for k, v in mac.items():
                if k == "commodity":
                    st.markdown("**上下游商品价格（行业相关）**")
                    if rel.get("detail"):
                        st.caption(rel["detail"])
                    if rel_items:
                        _show_df(pd.DataFrame([
                            {"商品": nm, "最新价": _r((d or {}).get("price")),
                             "涨跌幅%": _r((d or {}).get("pct_chg")),
                             "日期": str((d or {}).get("date") or "")[:10]}
                            for nm, d in rel_items.items()]))
                    else:
                        st.caption("（本次未取到相关商品价格）")
                    continue
                if k == "commodity_relevant":
                    continue          # 已在上面合并展示
                if k == "industry_names":
                    if v:
                        st.markdown("**所属行业/概念**")
                        st.caption("、".join(str(x) for x in v))
                    continue
                if isinstance(v, dict) and v:
                    st.markdown("**%s**" % mac_label.get(k, k))
                    if k == "market_regime":
                        st.caption("大盘水位影响所有个股的**系统性风险**，"
                                   "是长周期预测的重要先验。")
                    _render_section(v, label_map=mac_label)
        else:
            st.warning("未取得宏观数据（依赖 akshare 宏观接口，受限时用配置默认值兜底）。")
        _factor_block(p, "宏观/行业", title="该维度生成的因子")

    with tab7:
        news = p.get("news") or []
        if news:
            rel_kw = (p.get("name") or "").strip()
            code_kw = (p.get("symbol") or "").strip()
            rows = []
            for n in news:
                d = n.get("direction", 0)
                title = str(n.get("title", ""))
                related = (not rel_kw or rel_kw in title or code_kw in title)
                rows.append({"日期": str(n.get("date", ""))[:10],
                             "标题": title[:70],
                             "标签": n.get("tag", "") or n.get("type", ""),
                             "倾向": "利好" if d and d > 0 else ("利空" if d and d < 0 else "中性"),
                             "相关": "是" if related else "否",
                             "来源": n.get("source", "")})
            _show_df(pd.DataFrame(rows))
            st.caption("共 %d 条。标注「相关」表示标题中直接出现该股名称或代码；"
                       "标注为「是」的是与个股直接相关的资讯/公告。" % len(rows))
        else:
            st.warning("未取得近期新闻/公告。个股新闻依赖东财个股资讯接口、"
                       "公告依赖东财公告接口，若网络受限则为空。")
        _factor_block(p, "消息/事件", title="该维度生成的因子")

    with tab8:
        if sea:
            sea_label = {
                "date": "评估日期", "month": "当前月份", "weekday": "星期",
                "score": "综合评分", "calendar": "日历效应", "month_effect": "月份效应",
                "month": "月份效应", "weekday_effect": "星期效应", "holiday": "长假效应",
                "holiday_effect": "长假效应", "earnings": "财报窗口",
                "earnings_window": "财报窗口", "two_sessions": "两会窗口",
                "month_end": "月末效应", "fib_window": "斐波那契时间窗",
                "gann_cycle": "江恩时间周期", "solar_terms": "二十四节气",
                "report": "汇总", "consensus": "理论综合", "today": "当日",
                "symbol": None, "as_of": None, "notes": None,
            }
            # 标量字段全部展示（缺数据的显示为"—"，让用户看清覆盖范围）
            flat = {}
            for k, v in sea.items():
                if isinstance(v, (dict, list)):
                    continue
                lab = sea_label.get(k, _label(k))
                if lab is None:
                    continue
                flat[lab] = v
            if flat:
                _kv_grid(flat, include_missing=True)
            for k, v in sea.items():
                # sea_label 里显式登记为 None 的（symbol/as_of/notes）以及
                # 内部字段（下划线开头）都不展示
                if k in sea_label and sea_label[k] is None:
                    continue
                lab = sea_label.get(k)
                if lab is None:
                    lab = _label(k)
                if lab is None:
                    continue
                if isinstance(v, (dict, list)) and v:
                    st.markdown("**%s**" % _esc(lab))
                    _render_value(v)
        else:
            st.info("未取得季节性数据。")
        _factor_block(p, "周期/季节", title="该维度生成的因子")

    with tab9:
        if th:
            labels = {"dow": "道氏理论", "elliott": "艾略特波浪", "fib": "斐波那契",
                      "gann": "江恩理论", "vpt": "量价时空", "fan": "扇形原理",
                      "pct": "百分比回撤", "chaos": "混沌理论", "resonance": "多周期共振",
                      "consensus": "理论综合"}
            field_label = {
                "trend": "趋势", "primary_trend": "主要趋势", "secondary_trend": "次级趋势",
                "phase": "所处阶段", "score": "评分", "confidence": "置信度",
                "wave": "浪级", "direction": "方向", "detail": "说明",
                "current_zone": "当前区域", "nearest_support": "最近支撑",
                "nearest_resistance": "最近压力", "volume_stage": "量能阶段",
                "price_stage": "价格阶段", "match": "是否共振", "current_line": "当前角度线",
                "position": "位置", "nearest_level": "最近档位", "position_pct": "区间位置",
                "hurst": "Hurst指数", "fractal_dimension": "分形维数",
                "market_state": "市场状态", "sensitivity": "敏感性",
                "resonance": "是否共振", "resonance_direction": "共振方向",
                "strength": "强度", "bull_count": "看涨模型数", "bear_count": "看跌模型数",
                "hh_hl": "高点抬高+低点抬高", "lh_ll": "高点降低+低点降低",
                "volume_confirm": "量能验证", "fib_conformance": "波浪吻合度",
            }
            for k, v in th.items():
                if isinstance(v, dict) and v:
                    _lab = labels.get(k)
                    if _lab is None:
                        _lab = _label(k)
                    if _lab is None:
                        continue
                    st.markdown("**%s**" % _esc(_lab))
                    _render_section(v, label_map=field_label)
                elif isinstance(v, (list, tuple)) and v:
                    _lab = labels.get(k)
                    if _lab is None:
                        _lab = _label(k)
                    if _lab is None:
                        continue
                    st.markdown("**%s**" % _esc(_lab))
                    _render_value(v)
                elif isinstance(v, (str, int, float, bool)) and v is not None:
                    st.write("%s：%s" % (_label(k), _fmt_value(k, v)))
        else:
            st.info("未取得理论模型结果。")
        _factor_block(p, "理论模型", title="该维度生成的因子")

    if stats.get("by_category"):
        st.markdown("---")
        st.markdown("**因子分布**")
        rows = [{"类别": k, "看涨": v.get("看涨", 0), "看跌": v.get("看跌", 0),
                 "中性": v.get("中性", 0), "合计": sum(v.values())}
                for k, v in stats["by_category"].items()]
        _show_df(pd.DataFrame(rows))


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


def _kv_grid(d: dict, *, include_missing: bool = False) -> None:
    """底层卡片网格（**推荐直接用 :func:
ender_fields**）。

    保留它是为了少数"键已经是中文、值已经格式化好"的场景；
    注意：dict/list 值会被跳过（请改用 render_fields，它会递归展开而不是丢数据）。
    """
    items = []
    for k, v in d.items():
        lab = _label(k)
        if lab is None or isinstance(v, (dict, list)):
            continue
        missing = (v is None or (isinstance(v, float) and v != v)
                   or (isinstance(v, str) and v.strip() in ("", "nan", "None")))
        if missing and not include_missing:
            continue
        items.append((lab, v, missing))
    if not items:
        st.caption("（本次未取得该项数据）")
        return
    _kv_grid_items(items)


def _pct01(v):
    """把 0~1 的分位/占比转成百分数（0.0182 → "1.8%"）。"""
    try:
        f = float(v)
        if f != f:
            return "—"
        if 0.0 <= f <= 1.0:
            return "%.1f%%" % (f * 100)
        return "%.4g" % f
    except (TypeError, ValueError):
        return "—"


def _short(v) -> str:
    """把任意值转成简短中文显示（含枚举值翻译）。"""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        if v != v:
            return "—"
        if abs(v) >= 1e8:
            return _big(v)
        return "%.4g" % v
    if isinstance(v, int):
        return ("%d" % v) if abs(v) < 1e8 else _big(float(v))
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return "—"
    if s in _VALUE_LABELS:
        return _VALUE_LABELS[s]
    sl = s.lower()
    if sl in _VALUE_LABELS:
        return _VALUE_LABELS[sl]
    # 形如 "akshare.macro_china_cpi" 的数据源标识，不展示原始英文
    if "." in s and "/" not in s and " " not in s and len(s) < 40 \
            and any(p in sl for p in ("akshare", "sina", "eastmoney", "tencent",
                                      "proxy.finance", "qt.gtimg")):
        return "网络接口"
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


def render_self_learning(p: dict, symbol: str) -> None:
    """展示自我学习状态：历史核账结果、校准系数、维度权重。"""
    st.markdown("### 自我学习（程序随使用不断校准自己）")
    rep = p.get("learning_report") or {}
    info = p.get("learning") or {}
    rec = p.get("recorded") or {}

    if not rep and not info:
        st.info("自我学习未启用或暂无数据。可在左侧「分析设置」中开启「启用自我学习」。")
        return

    st.markdown(
        "<div class='tip'>这个程序的预测会**被自己记账并回头核对**：每次预测都写入档案，"
        "周期走完后用真实价格核账，算出「说 60%% 涨的时候实际对了多少」，"
        "再用这个结果反过来校准概率与置信度。因此**用得越久，它对这只股票越了解**。<br>"
        "样本不足时会先用历史数据回填一批虚拟预测（严格因果），避免新用户干等几个月。</div>",
        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("已核账样本", rep.get("evaluated", 0))
    c2.metric("待核账（周期未走完）", rep.get("pending", 0))
    c3.metric("本次新增记录", (rec or {}).get("rows", "—"))
    c4.metric("上次学习", (rep.get("last_update") or "—")[:10])

    if rep.get("detail"):
        st.caption(rep["detail"])

    # 冷启动回填说明
    bf = info.get("backfill") or {}
    if bf.get("detail"):
        st.caption("冷启动回填：%s" % bf["detail"])

    table = rep.get("table") or []
    if table:
        st.markdown("**各周期核账结果与校准强度**")
        _show_df(pd.DataFrame(table))
        st.caption("「方向命中率」= 该周期历史上预测方向对的比例；"
                   "「置信度乘子」= 学习后对置信度的放大/缩小（>1 更有把握、<1 更保守）；"
                   "「生效程度」= 样本量决定的收缩系数（0 表示样本太少、暂不生效，1 表示完全生效）。")

    dw = rep.get("dimension_weights") or {}
    if dw:
        st.markdown("**学到的维度权重（哪个维度的因子对这只股票真正有效）**")
        _kv_grid_items([(_label(k), "×%.2f" % v, False)
                        for k, v in dw.items() if _label(k)])
        st.caption("权重 >1 表示该维度对这只股票的历史预测更有效，<1 表示效果较差。")
    else:
        st.caption("维度权重尚未形成：它需要**真实预测**（带因子快照）积累到 20 条以上 20 日周期的"
                   "核账样本才会生效——历史回填只校准概率与置信度，不含当时的因子快照。"
                   "继续使用本程序，这部分会随真实使用逐步建立。")

    dec = info.get("learn") or {}
    if isinstance(dec, dict) and dec.get("by_symbol"):
        with st.expander("查看学习明细（原始校准参数）"):
            st.json(dec)


def render_backtest(p: dict, do_backtest: bool, symbol: str, days: int, demo: bool) -> None:
    if not do_backtest:
        st.info("已在侧边栏关闭历史回测。")
        return
    # ---- 自我学习：先展示"程序从历史核账中学到了什么" ----
    render_self_learning(p, symbol)
    st.markdown("---")
    st.markdown("### 历史回测（当前信号的样本外表现）")
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
                _show_df(tbl)
                st.markdown("</div>", unsafe_allow_html=True)
            fig = charts_mod.plot_backtest(tbl, grade)
            if fig is not None:
                import io

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
                st.image(buf.getvalue(), width="stretch")
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
        c1.download_button("下载 HTML 报告", html_txt.encode("utf-8"),
                           file_name="%s_预测报告.html" % symbol, mime="text/html",
                           width="stretch")
    if txt_txt:
        c2.download_button("下载文本报告", txt_txt.encode("utf-8"),
                           file_name="%s_预测报告.txt" % symbol, mime="text/plain",
                           width="stretch")
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
    if st.button("数据源健康检查", width="content"):
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

    # 程序名称统一放进顶部渐变横幅（不再单独放一个贴顶的 ## 标题，
    # 否则页面滚动时标题会被视口顶边裁掉一半，看起来像"显示不全"）
    st.markdown(
        "<div class='hero appbanner'>"
        "<h1>个股涨跌概率预测模型 V1.0</h1>"
        "<div class='meta'>技术面 · 基本面 · 资金面 · 情绪面 · 宏观 · 消息面 · 周期 · "
        "金融理论　→　未来 1天 / 5天 / 10天 / 20天 / 1个月 / 1个季度 / 半年 / 1年 "
        "的涨跌概率与幅度推断</div>"
        "<div class='byline'>田夫 开发</div>"
        "</div>", unsafe_allow_html=True)

    sym_raw = (opts["symbol"] or "").strip()
    if not sym_raw:
        st.info("请在左侧输入股票代码，例如 `600519`、`000001`、`300750`。")
        st.markdown("<div class='risk'>%s</div>" % DISCLAIMER, unsafe_allow_html=True)
        return

    # 起步就校验代码能否识别，避免用户输入中文名后拿不到数据却看不出原因
    try:
        _info = normalize_symbol(sym_raw)
        if not _info.get("code"):
            # 尝试按名称/拼音解析（单次联想请求，很快）
            with st.spinner("正在按名称查找股票…"):
                try:
                    from core.engine import lookup_code_by_name

                    _hits = lookup_code_by_name(sym_raw, limit=8)
                except Exception:  # noqa: BLE001
                    _hits = []
            if not _hits:
                st.warning(
                    "无法识别「%s」。可以试试：\n\n"
                    "* 直接输入 **6 位代码**：`600519`（贵州茅台）、`000001`（平安银行）、"
                    "`300750`（宁德时代）\n"
                    "* 支持 `sh600519`、`000001.SZ`、`00700.HK`、`AAPL` 等写法\n\n"
                    "按名称搜索依赖行情网站的联想接口，"
                    "该接口在部分网络环境下可能不可用。" % sym_raw)
                return
            opts_list = ["%s（%s）" % (h["name"] or h["code"], h["code"]) for h in _hits]
            picked = st.radio(
                "找到多个匹配，请选择要分析的股票：", opts_list,
                index=0, horizontal=False, key="name_pick",
                # 用回调写 symbol_input，避免"部件实例化后不可修改"的错误
                on_change=_apply_symbol,
                args=(_hits[0]["code"],))
            # 若用户切换了单选项，按当前选择重新解析
            try:
                _idx = opts_list.index(picked)
            except ValueError:
                _idx = 0
            sym_raw = _hits[_idx]["code"]
            st.caption("已解析为代码 **%s**，下方结果对应这只股票。" % sym_raw)
    except Exception as exc:  # noqa: BLE001
        st.warning("代码解析失败：%s" % exc)
        return

    if not st.session_state.get("run_analysis"):
        st.info("请在左侧输入代码后点击「开始分析」。")
        st.markdown("<div class='risk'>%s</div>" % DISCLAIMER, unsafe_allow_html=True)
        return

    prog = st.progress(0.0, text="准备中…")
    _stage_names = {
        "获取行情数据": "① 获取行情与基本面数据",
        "八大维度分析": "② 八大维度因子分析（技术/资金/情绪/宏观/消息/周期/理论）",
        "因子汇总与概率推导": "③ 因子汇总与概率推导",
        "训练AI模型": "④ 训练 AI 模型",
        "推导概率": "⑤ 推导各周期概率与幅度",
        "生成报告": "⑥ 生成报告",
    }

    def _on_progress(stage: str, pct: float) -> None:
        try:
            label = _stage_names.get(stage, stage)
            prog.progress(min(max(float(pct), 0.0), 1.0),
                          text="%s ｜ %.0f%%" % (label, float(pct) * 100))
        except Exception:  # noqa: BLE001
            pass

    try:
        _on_progress("获取行情数据", 0.02)
        payload = cached_analysis(sym_raw, bool(opts["use_ml"]), bool(opts["demo"]),
                                  int(opts["days"]), bool(opts.get("self_learning", True)),
                                  _progress_cb=_on_progress)
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

    tabs = st.tabs([":material/insights: 预测总览", ":material/candlestick_chart: 图表",
                    ":material/search: 因子与逻辑", ":material/explore: 八维明细",
                    ":material/science: 历史验证", ":material/download: 导出/工具"])

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
                st.caption("未完成：%s" % e)
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
