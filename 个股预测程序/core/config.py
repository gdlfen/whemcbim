# -*- coding: utf-8 -*-
"""
全局配置：因子权重矩阵、指标参数、阈值、数据源与运行参数。

所有"可调"的东西都集中在这里，便于针对不同市场状态做校准，也便于用户自行修改。

权重体系说明
------------
某周期下某因子的最终权重按以下顺序解析（先命中者优先）：

1. ``FACTOR_WEIGHTS[horizon_key][factor_id]`` —— 单因子精调（最优先）
2. ``FACTOR_WEIGHTS[horizon_key][category]``  —— 该周期下的大类权重
3. ``CATEGORY_WEIGHTS[kind][category]``       —— 该周期类型下的大类权重
4. ``DEFAULT_CATEGORY_WEIGHT``                —— 兜底

设计理念（"周期决定话语权"）：

* **短线（1~10 天）**：资金面 ≈ 技术面 > 情绪 > 消息 > 理论 > 基本面 > 宏观 > 周期
* **中线（20 天~1 季度）**：技术面 ≈ 基本面 ≈ 资金面 > 理论 > 情绪 > 宏观/消息 > 周期
* **长线（半年~1 年）**：基本面 ≈ 宏观 > 理论 > 资金面 > 周期 > 技术面 > 情绪 > 消息
"""

from __future__ import annotations

import json
import os

# --------------------------------------------------------------------------- #
# 运行参数
# --------------------------------------------------------------------------- #
RUNTIME = {
    "cache_dir": os.path.join(os.path.expanduser("~"), ".stock_predictor_cache"),
    "cache_ttl_seconds": 6 * 3600,      # 行情/快照缓存有效期
    "cache_ttl_daily": 24 * 3600,       # 财务/股东等低频数据缓存有效期
    "http_timeout": 12.0,
    "http_retries": 3,
    "http_backoff": 0.6,
    "user_agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "max_workers": 8,                   # 并行取数线程数
    "offline": False,                   # True 时不发网络请求，只用缓存/合成数据
    "log_level": "INFO",
    "history_days": 3200,               # 期望拉取的日线根数
    "index_symbol": "sh000001",         # 基准指数（上证指数）
    "index_code": "000001",
    "risk_free_rate": 0.018,            # 无风险利率（年化，用于夏普/超额）
    "trading_days_per_year": 244,
    "min_history_for_ml": 260,          # 训练机器学习模型所需最少样本
    "ml_min_samples": 120,              # 单周期训练最少样本
}

# --------------------------------------------------------------------------- #
# 技术指标参数
# --------------------------------------------------------------------------- #
INDICATOR_PARAMS = {
    "ma": (5, 10, 20, 30, 60, 120, 250),
    "ema": (12, 26, 50),
    "expma": (12, 50),
    "macd": (12, 26, 9),
    "kdj": (9, 3, 3),
    "rsi": (6, 12, 14, 24),
    "boll": (20, 2.0),
    "ene": (25, 11, 9),          # 轨道线 ENE：UPPER=(1+11%)*MA25, LOWER=(1-9%)*MA25
    "sar": (4, 2, 20),           # af_step=0.02, af_max=0.2, init=0.02
    "dkx": (10, 10),             # 多空线：MID=(3*C+L+O+H)/6, DKX=MA(MID,10)
    "bbi": (3, 6, 12, 24),
    "bias": (6, 12, 24),
    "cci": (14,),
    "dmi": (14, 6),
    "wr": (6, 10, 14),
    "dma": (10, 50, 10),
    "trix": (12, 20),
    "brar": (26,),
    "vr": (26,),
    "obv": (30,),
    "asi": (26,),
    "emv": (14, 9),
    "wvad": (24,),
    "roc": (12, 6),
    "lon": (10,),
    "psy": (12,),
    "mtm": (12, 6),
    "cr": (26,),
    "vwap_window": 60,           # 持仓成本线/筹码
    "chip_bins": 120,            # 筹码分布价格分箱数
    "fib_ratios": (0.191, 0.236, 0.382, 0.5, 0.618, 0.786, 0.809),
    "gann_ratios": (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875),
    "time_window_bars": (144, 89, 55, 34, 21, 13, 8, 5),   # 斐波那契时间窗
    "sr_lookback": 250,          # 支撑压力位回溯根数
    "sr_cluster_pct": 0.02,      # 价格聚类带宽（2%）
}

# --------------------------------------------------------------------------- #
# 预测/概率参数
# --------------------------------------------------------------------------- #
SPEED = {
    "parallel": True,          # 是否并行执行分析阶段（关闭便于调试）
    # 并发与性能（在保证数据质量的前提下加速）
    "board_workers": 8,        # 板块历史K线并发线程数
    "market_workers": 6,       # 大盘指数/板块列表并发线程数
    "board_max": 24,           # 最多抓多少个板块的历史（太多会拖慢）
    "index_workers": 6,        # global_market 指数历史并发数
    "min_interval_s": 0.05,    # 同一域名的请求最小间隔（避免被封）
}

FUND = {
    # 场内基金（ETF/LOF/封基/REITs）专属参数
    "risk_free_rate": 0.02,          # 夏普比率用的无风险利率
    "premium_warn_pct": 0.35,        # 折溢价率超过该值即提示（%）
    "premium_volatile_std": 0.50,    # 折溢价 20 日标准差超过该值视为定价效率低（%）
    "tracking_error_good": 2.0,      # 年化跟踪误差优于该值视为优秀（%）
    "size_clearance_yi": 0.5,        # 规模低于该值（亿元）提示清盘风险
    "size_large_yi": 50.0,           # 规模高于该值（亿元）视为充裕
    "nav_pages": 3,                  # 抓取净值历史页数（每页 200 条）
}

FORECAST = {
    "flat_band_pct": {           # 视为"横盘"的收益带宽（%），按周期
        "d1": 0.8, "d5": 1.8, "d10": 2.5, "d20": 3.5,
        "m1": 3.8, "q1": 7.0, "h1": 10.0, "y1": 15.0,
    },
    "big_move_pct": {            # "大涨/大跌"阈值（%），按周期
        "d1": 3.0, "d5": 6.0, "d10": 8.0, "d20": 12.0,
        "m1": 13.0, "q1": 22.0, "h1": 32.0, "y1": 50.0,
    },
    "score_to_prob_scale": {     # 综合得分 -> 概率的换算斜率（logit 系数）
        "d1": 2.05, "d5": 1.90, "d10": 1.75, "d20": 1.60,
        "m1": 1.45, "q1": 1.15, "h1": 1.00, "y1": 0.90,
    },
    "prior_p_up": 0.5,           # 先验上涨概率（A股长期略偏多，可设 0.505）
    "prob_clip": (0.05, 0.95),   # 概率输出截断，避免过度自信
    "ml_blend": {                # ML 与规则模型融合权重（ML 占比），按周期
        "d1": 0.30, "d5": 0.34, "d10": 0.36, "d20": 0.38,
        "m1": 0.38, "q1": 0.34, "h1": 0.30, "y1": 0.26,
    },
    "ml_enabled": True,
    # 自我学习：用历史核账记录校准概率与置信度（详见 core/learning.py）
    "self_learning": True,
    "ml_model": "auto",          # auto | logit | gbdt | none
    "theory_penalty": 0.15,      # 理论模型因主观性打折的系数
    "long_horizon_fundamental_boost": 1.25,   # 长周期基本面权重放大
    "high_vol_attenuation": 0.75,             # 高波动股方向判断的衰减系数
}

# --------------------------------------------------------------------------- #
# 数据源代码优先级（靠前者先试，失败自动降级）
# --------------------------------------------------------------------------- #
DATASOURCE_PRIORITY = {
    "kline_daily": ("sina", "tencent", "eastmoney", "akshare", "yfinance"),
    "kline_weekly": ("sina", "tencent", "eastmoney"),
    "kline_monthly": ("sina", "tencent", "eastmoney"),
    "realtime": ("tencent", "sina", "eastmoney"),
    "fund_flow": ("eastmoney", "akshare", "sina"),
    "financials": ("eastmoney", "akshare", "sina", "tencent"),
    "valuation": ("eastmoney", "tencent", "akshare"),
    "holders": ("eastmoney", "akshare"),
    "margin": ("eastmoney", "akshare"),
    "lhb": ("eastmoney", "akshare"),
    "northbound": ("eastmoney", "akshare"),
    "news": ("eastmoney", "sina", "tencent"),
    "index": ("sina", "eastmoney", "akshare", "tencent"),
    "sector": ("eastmoney", "akshare"),
    "macro": ("eastmoney", "akshare", "local"),
}

# --------------------------------------------------------------------------- #
# 因子权重矩阵
# --------------------------------------------------------------------------- #
DEFAULT_CATEGORY_WEIGHT = 0.5

# 周期类型 -> 大类基础权重（相对值，引擎内部会归一化）
CATEGORY_WEIGHTS = {
    "short": {          # 1~10 个交易日：资金与情绪主导
        "technical": 1.00,
        "capital": 1.05,
        "sentiment": 0.80,
        "news": 0.70,
        "theory": 0.55,
        "stage": 0.60,
        "fundamental": 0.35,
        "macro": 0.30,
        "cycle": 0.40,
        "ml": 0.90,
    },
    "medium": {         # 20 个交易日 ~ 1 个季度：技术+基本面+资金三足鼎立
        "technical": 1.00,
        "fundamental": 0.95,
        "capital": 0.90,
        "theory": 0.75,
        "sentiment": 0.60,
        "news": 0.55,
        "macro": 0.60,
        "cycle": 0.55,
        "stage": 0.65,
        "ml": 0.85,
    },
    "long": {           # 半年 ~ 1 年：基本面与宏观决定水位
        "fundamental": 1.15,
        "macro": 1.00,
        "theory": 0.85,
        "capital": 0.70,
        "cycle": 0.70,
        "stage": 0.75,
        "technical": 0.60,
        "sentiment": 0.45,
        "news": 0.40,
        "ml": 0.60,
    },
}

# 单周期精调（按 factor id 或 category 名）；留空则用上面的大类权重
FACTOR_WEIGHTS: dict[str, dict[str, float]] = {
    "d1": {
        "technical": 1.10, "capital": 1.25, "sentiment": 0.95, "ml": 1.00,
        "cycle": 0.35, "fundamental": 0.20, "macro": 0.20,
        # 单因子精调
        "macd_cross": 1.15, "kdj_cross": 1.20, "vol_price": 1.25,
        "capital_main_net": 1.35, "capital_big_order": 1.30,
        "quant_activity": 1.15, "limit_up_strength": 1.25,
        "gap_pattern": 1.20, "sr_position": 1.20, "sar_flip": 1.10,
        "fundamental_growth": 0.15, "valuation_percentile": 0.15,
    
            "fundspec": 0.85,},
    "d5": {
        "technical": 1.10, "capital": 1.20, "sentiment": 0.90, "ml": 0.95,
        "fundamental": 0.35, "macro": 0.30, "news": 0.80,
        "macd_cross": 1.10, "ma_alignment": 1.15, "capital_main_net": 1.30,
        "sector_heat": 1.15, "theory_elliott": 0.70,
    
            "fundspec": 0.85,},
    "d10": {
        "technical": 1.05, "capital": 1.10, "sentiment": 0.85, "ml": 0.90,
        "fundamental": 0.45, "macro": 0.40, "news": 0.70,
    
            "fundspec": 0.80,},
    "d20": {
        "technical": 1.00, "fundamental": 0.80, "capital": 0.95, "theory": 0.80,
        "macro": 0.55, "cycle": 0.60, "news": 0.65, "ml": 0.85,
        "ma_alignment": 1.10, "boll_position": 1.05,
    
            "fundspec": 0.75,},
    "m1": {
        "technical": 1.00, "fundamental": 0.90, "capital": 0.90, "theory": 0.80,
        "macro": 0.60, "cycle": 0.65, "news": 0.60, "ml": 0.85,
    
            "fundspec": 0.75,},
    "q1": {
        "technical": 0.85, "fundamental": 1.05, "macro": 0.85, "theory": 0.85,
        "capital": 0.75, "cycle": 0.70, "stage": 0.70, "ml": 0.80,
        "fundamental_growth": 1.30, "industry_policy": 1.20, "earnings_catalyst": 1.15,
    
            "fundspec": 0.70,},
    "h1": {
        "fundamental": 1.20, "macro": 1.05, "theory": 0.85, "cycle": 0.75,
        "capital": 0.65, "technical": 0.55, "sentiment": 0.40, "ml": 0.60,
        "fundamental_growth": 1.35, "valuation_percentile": 1.30, "roe_quality": 1.25,
    
            "fundspec": 0.65,},
    "y1": {
        "fundamental": 1.35, "macro": 1.10, "theory": 0.80, "cycle": 0.80,
        "capital": 0.55, "technical": 0.45, "sentiment": 0.35, "news": 0.30, "ml": 0.55,
        "fundamental_growth": 1.40, "valuation_percentile": 1.40, "roe_quality": 1.30,
        "industry_policy": 1.25, "macro_cycle": 1.20,
        # 按行业筛选后的商品价格比"全品类商品"更贴合个股，给更高权重
        "commodity_cost_industry": 0.85, "commodity_cost": 0.30,
    
            "fundspec": 0.60,},
}

# 反向指标：因子看涨时实际偏空（用于散户情绪等），引擎读取后自动翻转
CONTRARIAN_FACTOR_IDS = {
    "retail_sentiment_contrarian",
    "guba_bull_ratio",
    "search_heat_spike",
    "turnover_euphoria",
    "margin_euphoria",
    "limit_up_frenzy",
}

# --------------------------------------------------------------------------- #
# 阶段判定阈值（个股所处阶段）
# --------------------------------------------------------------------------- #
STAGE_RULES = {
    "trend_ma_slope_bars": 20,        # 判趋势用的均线斜率窗口
    "strong_trend_adx": 25.0,
    "sideways_bb_width_pct": 0.08,    # 布林带宽度 < 8% 视为横盘
    "high_vol_annual_pct": 55.0,      # 年化波动率 > 55% 视为高波动
    "low_vol_annual_pct": 22.0,
    "volume_surge_ratio": 2.0,        # 量能异动倍数
    "limit_up_pct": {"主板": 9.8, "创业板": 19.5, "科创板": 19.5, "北交所": 29.5, "ST": 4.8},
    "quant_turnover_high_pct": 8.0,   # 换手率 > 8% 且振幅大 → 量化活跃度高
    "quant_amp_high_pct": 4.5,
    "pricing_power_top10_pct": 55.0,  # 前十大流通股东占比 > 55% → 机构/主力定价
    "pricing_power_retail_pct": 25.0, # < 25% → 散户/游资定价特征
}

# --------------------------------------------------------------------------- #
# 宏观指标（本地默认值 + 网络覆盖）
# --------------------------------------------------------------------------- #
MACRO_DEFAULTS = {
    "gdp_yoy": 5.0,
    "cpi_yoy": 0.5,
    "ppi_yoy": -1.5,
    "pmi": 50.0,
    "lpr_1y": 3.0,
    "lpr_5y": 3.5,
    "shibor_3m": 1.7,
    "m2_yoy": 8.0,
    "social_financing_yoy": 8.5,
    "usd_cny": 7.10,
    "vix": 16.0,
}

# 政策/行业关键词 → 影响的题材与方向（消息面打分用）
POLICY_KEYWORDS = {
    "利好": ["补贴", "减税", "税收优惠", "降准", "降息", "扩容", "试点", "国产替代",
             "以旧换新", "专项债", "重大项目", "采购", "中标", "订单", "重组", "并购",
             "回购", "增持", "股权激励", "分红", "提价", "涨价", "获批", "通过审批",
             "放量", "满产", "扩产", "景气", "超预期", "预增", "扭亏"],
    "利空": ["减持", "清仓", "违规", "处罚", "立案", "调查", "问询", "诉讼", "仲裁",
             "质押", "冻结", "退市", "商誉减值", "计提", "预亏", "下滑", "终止",
             "解禁", "限售", "反倾销", "制裁", "禁令", "限产", "停产", "召回",
             "降价", "内卷", "产能过剩"],
}

# 题材热度关键词 → 概念板块（用于判断是否处于风口）
HOT_THEMES = [
    "AI", "人工智能", "算力", "大模型", "机器人", "人形机器人", "半导体", "芯片",
    "国产替代", "新能源", "光伏", "储能", "锂电池", "新能源车", "智能驾驶", "低空经济",
    "商业航天", "卫星互联网", "军工", "国企改革", "中特估", "数据要素", "信创", "华为",
    "医药", "创新药", "减肥药", "白酒", "消费复苏", "房地产", "券商", "银行", "黄金",
    "有色金属", "稀土", "核电", "可控核聚变", "量子科技", "固态电池", "脑机接口",
]

# 日历效应先验（月 -> 历史平均月收益的经验值，A股口径，用于弱先验）
MONTH_EFFECT_PRIOR = {
    1: 0.35, 2: 1.10, 3: 0.55, 4: -0.65, 5: -0.40, 6: -0.45,
    7: 0.25, 8: -0.20, 9: -0.15, 10: 0.55, 11: 0.45, 12: 0.60,
}

# 输出与展示
OUTPUT = {
    "report_dir": "reports",
    "chart_dir": "charts",
    "keep_charts_days": 30,
}

_cache_path_checked = False


def ensure_cache_dir() -> str:
    """确保缓存目录存在并返回其路径。"""
    d = RUNTIME["cache_dir"]
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")
        os.makedirs(d, exist_ok=True)
    return d


def category_weights(horizon_key: str, kind: str) -> dict:
    """返回某周期下的大类权重字典（未归一化）。"""
    base = dict(CATEGORY_WEIGHTS.get(kind, CATEGORY_WEIGHTS["medium"]))
    override = FACTOR_WEIGHTS.get(horizon_key, {})
    for k, v in override.items():
        # 只把"不是大类名"的键当单因子处理；大类名直接覆盖
        base[k] = v
    return base


def factor_weight(horizon_key: str, kind: str, factor_id: str, category: str) -> float:
    """解析某因子在某周期下的权重。"""
    per_h = FACTOR_WEIGHTS.get(horizon_key, {})
    if factor_id in per_h:
        return float(per_h[factor_id])
    if category in per_h:
        return float(per_h[category])
    base = CATEGORY_WEIGHTS.get(kind, CATEGORY_WEIGHTS["medium"])
    return float(base.get(category, DEFAULT_CATEGORY_WEIGHT))


def dump_config(path: str) -> str:
    """把当前配置导出为 JSON（便于外部修改后回读）。"""
    data = {
        "RUNTIME": RUNTIME,
        "INDICATOR_PARAMS": {k: list(v) if isinstance(v, tuple) else v
                             for k, v in INDICATOR_PARAMS.items()},
        "FORECAST": FORECAST,
        "CATEGORY_WEIGHTS": CATEGORY_WEIGHTS,
        "FACTOR_WEIGHTS": FACTOR_WEIGHTS,
        "STAGE_RULES": STAGE_RULES,
        "MACRO_DEFAULTS": MACRO_DEFAULTS,
        "DATASOURCE_PRIORITY": {k: list(v) for k, v in DATASOURCE_PRIORITY.items()},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":  # pragma: no cover
    from contracts import HORIZONS  # type: ignore

    for h in HORIZONS:
        w = category_weights(h.key, h.kind)
        top = sorted(w.items(), key=lambda kv: -kv[1])[:4]
        print("%-4s %-16s kind=%-6s 主导: %s" % (
            h.key, h.label, h.kind, ", ".join("%s=%.2f" % kv for kv in top)))
    print("config self-test OK")
