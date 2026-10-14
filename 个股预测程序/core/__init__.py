# -*- coding: utf-8 -*-
"""
个股涨跌概率预测程序 —— 核心计算库。

模块地图
--------
``contracts``   数据契约：:class:`Factor` / :class:`AnalysisBundle` / :class:`ProbabilityForecast`
``config``      权重矩阵、指标参数、阈值、数据源优先级
``utils``       数值/概率工具、HTTP、缓存、代码规范化
``datasource``  多源容错取数（日线/周线/实时/指数/板块）
``indicators``  40+ 技术指标（MA/EMA/MACD/KDJ/RSI/BOLL/ENE/SAR/DKX/BBI/EXPMA/BIAS/
                CCI/DMI/WR/DMA/TRIX/BRAR/VR/OBV/ASI/EMV/WVAD/ROC/LON/PSY/CR/筹码…）
``patterns``    K 线经典形态识别（启明星/黄昏星/吞没/锤子/三只乌鸦/缺口/岛形…）
``technical``   技术面因子合成（多周期、量价、背离、支撑压力、通道箱体、黄金分割）
``fundamental`` 基本面因子（成长/质量/现金流/负债/估值分位/PEG/催化剂）
``capital``     资金面与主力动向（主力净额/大单/龙虎榜/北向/股东户数/控盘度）
``sentiment``   情绪与博弈（两融/换手狂热/板块热度/散户反向指标）
``macro``       宏观与行业（利率/CPI/PMI/政策/上下游）
``newsflow``    消息面与突发事件（公告分类打分、黑天鹅识别）
``seasonality`` 日历效应与技术周期（财报窗口/两会/春节/月末/斐波那契时间窗/江恩）
``theory``      金融理论模型（道氏/艾略特波浪/斐波那契/江恩/量价时空/扇形/百分比回撤）
``mlmodel``     AI 学习器（纯 numpy 逻辑回归 + 梯度提升 + 滚动前向训练）
``forecast``    多周期概率预测引擎（1天/5天/10天/20天/1月/1季/半年/1年）
``backtest``    历史回测与模型校准
``charts``      可视化
``report``      文本 / HTML 报告
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]


def __getattr__(name: str):
    """惰性导出常用符号，避免导入整个包时加载全部依赖。"""
    if name in ("Factor", "AnalysisBundle", "ProbabilityForecast", "PredictionResult",
                "HORIZONS", "horizon_spec", "FactorSet", "Category"):
        from . import contracts
        return getattr(contracts, name)
    if name in ("config", "utils", "datasource", "indicators", "patterns", "technical",
                "fundamental", "capital", "sentiment", "macro", "newsflow",
                "seasonality", "theory", "mlmodel", "forecast", "backtest",
                "charts", "report", "engine"):
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
