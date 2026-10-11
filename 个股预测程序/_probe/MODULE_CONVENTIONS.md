# 五个分析模块的统一实现约定（写代码前必读）

> 本文件由主控 agent 编写，是 `core/fundamental.py`、`core/capital.py`、`core/sentiment.py`、
> `core/macro.py`、`core/newsflow.py` 五个模块必须共同遵守的实现细则。
> 冻结文件（**禁止修改**）：`core/contracts.py`、`core/utils.py`、`core/config.py`、
> `core/selftest_data.py`、`core/__init__.py`、`DEVELOPER_BRIEF.md`。

## 1. 环境事实（实测）

* Python 3.13.12 / Windows；`pandas 3.0.1`、`numpy 2.3.5`、`akshare 1.18.47`。
* **没有 scipy / sklearn / talib**。需要统计量时用 `core.utils` 里的
  `mean/stdev/percentile_rank/norm_cdf/norm_ppf/sigmoid/tanh_scale`，或 numpy 手写。
* pandas 3 注意点：不要用 `DataFrame.append` / `applymap` / `Series.iteritems`；
  用 `.map()`；不要依赖链式赋值（copy-on-write 默认开启，赋值用 `df.loc[:, col] = ...`）；
  `df.fillna` 对 object 列仍可用但建议显式 `inplace=False` 赋值。
* 运行命令一律：`$env:PYTHONIOENCODING='utf-8'; python -m core.xxx`，工作目录
  `E:\deepseek desktop\个股预测程序`。

## 2. 文件头与导入模板

```python
# -*- coding: utf-8 -*-
"""
模块 docstring：本模块负责哪些因素、产出哪些因子 id、数据源清单与降级策略。
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from . import config
from .contracts import (Category, DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Factor,
                        clamp, now_cn, safe_float)
from .utils import (cache_key, cached, get_logger, http_get, http_get_many,
                    normalize_symbol, percentile_rank, retry_call, stdev, mean)

log = get_logger("stockai.<模块名>")

__all__ = [...]
```

* 一律使用**相对导入**（`from . import config`），保证 `python -m core.xxx` 与包导入都可用。
* 网络函数**必须**包在 `try/except Exception` 中，失败 `log.debug(...)` 后返回
  `pd.DataFrame()` / `{}` / `[]`，**绝不 raise**。

## 3. 通用小工具（各模块内自行实现，避免跨模块耦合）

```python
def _ensure_frame(df):
    """列名容错：把大写/中文列名映射到小写英文标准列；不是 DataFrame 则返回 None。"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    ren = {"日期": "date", "时间": "date", "Date": "date", "DATE": "date", "trade_date": "date",
           "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
           "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_chg", "换手率": "turnover",
           "Open": "open", "Close": "close", "High": "high", "Low": "low", "Volume": "volume"}
    out = df.rename(columns={k: v for k, v in ren.items() if k in df.columns}).copy()
    return out


def _num(x, default=float("nan")):
    """安全取数（复用 contracts.safe_float，并额外兼容 numpy 标量）。"""
    v = safe_float(x, default)
    return v


def _last_valid(series, default=float("nan")):
    """取序列中最后一个非 NaN 值。"""
```

* **禁止修改调用方传入的 DataFrame**（统一 `.copy()`）。
* 数值阈值尽量取 `config` 里的常量（`STAGE_RULES` / `FORECAST` / `MACRO_DEFAULTS`），
  模块内再定义自己的常量时集中放在文件顶部并写注释。

## 4. 缓存约定

```python
from .utils import cache_key, cached, cache_ttl  # cache_ttl 不存在，用 config.RUNTIME
key = cache_key("<前缀>", code, ...)          # 前缀：fin / val / flow / holder / margin /
                                              # lhb / north / top10 / macro / news / ann / sector
df = cached(key, config.RUNTIME["cache_ttl_daily"], _producer)   # 低频数据（财务/股东/宏观）
df = cached(key, config.RUNTIME["cache_ttl_seconds"], _producer) # 行情/资金流/情绪
```

`force=True` 时绕过缓存直接取（内部判断 `if not force: cached(...) else producer()`）。
**缓存里不放 None**（`cached` 本身会跳过 None）；空 DataFrame 可以缓存。

## 5. Factor 生成约定

* 构造：`Factor(id, name, category, direction, strength, reliability, value=..., detail=..., theory=..., horizon=(...))`。
* `direction` ∈ {1,-1,0}，`strength`/`reliability` 会被自动 clamp 到 [0,1]。
* `horizon` 只能是 `"d1" "d5" "d10" "d20" "m1" "q1" "h1" "y1"` 的子集元组；`None` = 全周期。
* `detail` 必须是**给用户看的中文一句话**，带上关键数字（如 `"近 5 日主力净流入 3.2 亿，占成交额 4.1%"`）。
* `value` 放原始观测值（数字或短字符串）。
* `theory` 是理论标签，如 `"资金流向理论"`、`"筹码分布"`、`"行为金融学"`、`"DCF 估值"`、`"美林时钟"`。
* **reliability 标尺**（DEVELOPER_BRIEF 第 5 节）：
  数据完整且样本 ≥250 → 0.85~0.95；样本 60~250 或逻辑间接 → 0.60~0.80；
  样本 <60 或需二次推断 → 0.35~0.55；缺失用默认值 → 0.15~0.35；观点性强 → 再 ×0.8。
* **数据全空时也必须返回合理结果**：`frame=None, valuation=None, news=[]` 时，
  要么返回 `[]`，要么返回低 reliability（≤0.3）的中性/先验因子，**不得抛异常**。
* 因子生成函数一律 `factors: list[Factor] = []` 累加后 `return factors`。

## 6. 自测块要求（每个模块必须有）

```python
def _self_test() -> None:
    """离线自测：只用 core.selftest_data 合成数据 + monkeypatch 掉 fetch，绝不联网。"""
    ...

if __name__ == "__main__":  # pragma: no cover
    _self_test()
```

自测必须覆盖：
1. 用合成数据（`make_ohlcv` / `make_financials_frame` / `make_index` / `make_minute_flow`）
   构造输入，跑通所有**因子生成函数**，打印因子数量与 id 清单；
2. **数据全空用例**：`frame=None` 等，确认不抛异常；
3. 打印 `"<模块名> self-test OK"`（验收标准 1 要求字符串精确包含 `self-test OK`）。

## 7. 因子 id 清单（必须与 core/config.py 权重表 / CONTRARIAN_FACTOR_IDS 一致）

* **fundamental**：`fundamental_growth`、`growth_divergence`、`roe_quality`、`margin_trend`、
  `profit_quality`、`cashflow_health`、`leverage_risk`、`valuation_percentile`、`peg_level`、
  `relative_valuation`、`earnings_catalyst`、`turnaround`、`restructure_event`、
  `receivable_risk`、`loss_warning`。
* **capital**：`capital_main_net`、`capital_super_big`、`capital_main_ratio`、
  `capital_trend_consistency`、`capital_divergence`、`lhb_institution`、`northbound_change`、
  `holder_concentration`、`top10_float_change`、`fund_holding_change`、`margin_balance_trend`、
  `control_degree`、`main_force_behavior`。
* **sentiment**：`market_breadth`、`limit_up_frenzy`、`sector_heat`、`sector_leader_linkage`、
  `margin_euphoria`、`margin_trend`、`retail_sentiment_contrarian`、`guba_bull_ratio`、
  `search_heat_spike`、`turnover_euphoria`、`fear_greed`、`volatility_regime`。
* **macro**：`macro_cycle`、`interest_rate`、`inflation`、`liquidity`、`market_regime`、
  `industry_policy`、`regulation_risk`、`commodity_cost`、`supply_demand`、`fx_impact`。
* **newsflow**：`news_sentiment_agg`、`announcement_risk`、`buyback_incentive`、
  `restructure_event`、`earnings_preannounce`、`unlock_pressure`、`regulatory_risk`、
  `geopolitics`、`blackswan`。

> `core/config.py` 的 `CONTRARIAN_FACTOR_IDS` = {`retail_sentiment_contrarian`、`guba_bull_ratio`、
> `search_heat_spike`、`turnover_euphoria`、`margin_euphoria`、`limit_up_frenzy`}，
> 这些因子**按"原始看涨信号"写 direction**，由引擎翻转，**不要在模块里自己翻转**，
> 但 `detail` 要写清"这是反向指标"。

## 8. 真实数据验证要求（验收标准 3）

写完模块后，除了离线自测，还要跑一次真实数据（600519）并把关键输出写进最终报告：
优先用 `python -c "from core.xxx import ...; print(...)"` 的形式，
把返回的 DataFrame 前几行 / dict 的关键键值打印出来。
拿不到的接口要写明**失败原因**并确认已降级（返回空对象，不抛异常）。

## 9. 汇报格式

返回给主控的结果请包含：
1. 文件路径 + 公开函数清单（含签名）；
2. 离线自测的真实输出（`... self-test OK` 那一行以及因子数量/id 清单）；
3. 真实数据（600519）取到的字段样例；
4. 未能取到的数据源 + 降级方式；
5. 过程中做的任何与需求不同的**取舍**说明。
