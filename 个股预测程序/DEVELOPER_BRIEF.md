# 开发接口约定（子模块开发者必读）

## 0. 环境与硬性约束

* Python 3.13 / Windows；工作目录 `E:\deepseek desktop\个股预测程序`。
* 可用第三方库：`numpy`、`pandas`、`matplotlib`、`requests`、`akshare`、`yfinance`、`streamlit`。
  **不要引入 scipy / sklearn / talib / bottleneck / numba 等未安装库。**
* 所有文件 `# -*- coding: utf-8 -*-`，UTF-8 保存（含中文注释）。
* 文件路径全部为 ASCII，不要出现中文文件名。
* **绝不抛异常打断主流程**：任何函数拿不到数据/算不出结果时，返回空列表、`None`
  或降级结果，并（在需要时）通过 `core.utils.get_logger().debug(...)` 记录。
* 每个模块末尾必须有 `if __name__ == "__main__":` 自测块，使用
  `core.selftest_data` 的合成数据，**不联网**即可跑通，并打印 `xxx self-test OK`。

## 1. 统一数据契约（`core/contracts.py`，已实现，直接导入）

```python
from core.contracts import (
    Factor, FactorSet, AnalysisBundle, Category,
    DIR_BULL, DIR_BEAR, DIR_NEUTRAL,
    HORIZONS, HORIZON_KEYS, horizon_spec,
    ProbabilityForecast, PredictionResult, clamp, safe_float, pct_change, now_cn,
)
```

`Factor` 字段（全部为关键字友好的位置参数顺序）：

```python
Factor(id, name, category, direction, strength, reliability=0.6,
       value=None, detail="", theory="", horizon=None, weight_multiplier=1.0)
```

* `direction` ∈ {+1, -1, 0}；`strength`、`reliability` ∈ [0,1]（构造时自动 clamp）。
* `horizon=None` 表示对所有预测周期生效；否则传周期 key 元组，如 `("d1","d5","d10")`。
* `horizon` 取值只能是：`"d1" "d5" "d10" "d20" "m1" "q1" "h1" "y1"`。
* `category` 取 `Category.TECH / FUND / CAPITAL / SENTIMENT / MACRO / NEWS / CYCLE / THEORY / STAGE / ML`。
* `detail` 是一句给用户看的中文解释；`theory` 是理论依据标签（如 `"道氏理论"`、`"MACD"`、`"量价关系"`）。
* `value` 放原始观测值（数字或短字符串），会展示在报告里。

`AnalysisBundle` 的富数据字段（供报告/图表使用，取不到就留空）：
`ohlcv, kline_weekly, kline_monthly, indicators, snapshot, stage, valuation, financials,
capital, sentiment, macro, news, seasonality, theory, index_data, market, ml, charts, notes, errors`。
用 `bundle.note("...")` 记录数据源降级说明。

## 2. 其他可复用模块

```python
from core import config      # 权重矩阵/指标参数/阈值，见 core/config.py
from core.utils import (http_get, http_get_many, cached, cache_key, cache_get, cache_put,
                        normalize_symbol, board_of, limit_up_pct, get_logger,
                        sigmoid, logit, norm_cdf, norm_pdf, norm_ppf,
                        mean, stdev, percentile_rank, tanh_scale, fmt_num, fmt_pct,
                        retry_call, rolling_apply)
from core.selftest_data import make_ohlcv, make_index, make_minute_flow, make_financials_frame
```

`make_ohlcv(n=900, seed=42, start_price=30.0, start_date="2022-01-03", regime="mixed")`
返回 `DataFrame[date, open, high, low, close, volume, amount, pct_chg, turnover]`，
`date` 为 `pd.Timestamp`。`make_index(...)` 同结构（大盘）。`make_financials_frame(...)`
返回 `DataFrame[报告期, 营业收入, 营收同比, 净利润, 净利同比, ROE, 毛利率, 净利率,
资产负债率, 经营现金流, 每股收益]`。

`normalize_symbol("600519")` → `{"code","market","exchange","prefix","secid","yf_symbol","display","is_cn"}`；
`secid` 用于东财接口（`1.600519` / `0.000001`），`prefix` 为 `sh/sz/bj/hk/us`。

## 3. 数据分析约定

* 输入 K 线统一假设为 pandas DataFrame，列名小写：`date, open, high, low, close, volume, amount`。
  为稳健起见，函数内部应先 `df = _ensure_frame(df)` 做列名容错（大写/中文列名映射）。
* 指标函数返回 `pd.Series`，索引与输入一致；数据不足的头部返回 `NaN`（不 fillna、不裁剪长度）。
* 指标函数同时接受 `pd.Series` 与 `np.ndarray`：内部用 `s = pd.Series(np.asarray(x, dtype=float))` 兜底。
* **严禁**在函数内部修改传入的 DataFrame（用 `.copy()`）。

## 4. 已探测可用的数据源（供联网模块参考，均已验证）

| 用途 | 源 | 说明 |
|---|---|---|
| 日线（最长~3000根，2014年起） | Sina | `https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=sh600519&scale=240&ma=no&datalen=3000` → `[{"day","open","high","low","close","volume"}]`（不复权） |
| 日/周/月线（qfq/hfq） | Tencent | `https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh600519,day,,,640,qfq` → `data[code]["qfqday"]` = `[[date,open,close,high,low,volume], ...]`（注意顺序是 O/C/H/L/V，可能有第7列成交额） |
| 实时快照（含 PE/PB/市值/换手） | Tencent | `https://qt.gtimg.cn/q=sh600519`（GBK 编码，`~` 分隔） |
| 资金流（日/分钟） | 东财 | `https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get?secid=1.600519&fields1=f1,f2,f3,f7&fields2=f51,...,f65&klt=101&lmt=100` → `data.klines` 逗号分隔 |
| 十大流通股东 | 东财 | `https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_F10_EH_FREEHOLDERS&columns=ALL&filter=(SECUCODE="600519.SH")&pageSize=10&sortColumns=END_DATE&sortTypes=-1` |
| 两融明细 | 东财 | `reportName=RPTA_WEB_RZRQ_GGMX&filter=(SCODE="600519")` |
| 龙虎榜 | 东财 | `reportName=RPT_DAILYBILLBOARD_DETAILSNEW&filter=(SECURITY_CODE="600519")` |
| 概念板块资金流 | 东财 | `https://push2.eastmoney.com/api/qt/slist/get?secid=1.600519&spt=3&fields=f12,f13,f14,f3,f62,f184&po=1&pn=1&pz=30&fid=f62` |
| 个股新闻搜索 | 东财 | `https://search-api-web.eastmoney.com/search/jsonp?cb=cb&param={json}` |
| 财务摘要（80期指标） | akshare | `ak.stock_financial_abstract(symbol="600519")` |
| 指数日线（8000+根） | akshare | `ak.stock_zh_index_daily(symbol="sh000001")` |

东财 `push2`/`push2his` 域名的**普通行情**接口（kline/get、stock/get）在本机网络下会间歇性
`RemoteDisconnected`，必须用 `retry_call` 包 2~3 次；`datacenter-web` 与 `fflow` 稳定。
所有联网函数都必须能优雅降级，并保证离线可跑自测。

## 5. 因子可信度（reliability）建议标尺

| 情形 | reliability |
|---|---|
| 直接计算、数据完整、样本 ≥ 250 | 0.85 ~ 0.95 |
| 数据完整但样本 60~250，或逻辑间接 | 0.60 ~ 0.80 |
| 样本 < 60，或需二次推断 | 0.35 ~ 0.55 |
| 数据缺失、用默认值/先验替代 | 0.15 ~ 0.35 |
| 观点性强（政策解读、形态主观判断） | 再乘 0.8 |

## 6. 风格要求

* 中文注释、中文 `detail`，代码简洁可读，函数职责单一。
* 每个模块顶部写一段模块 docstring，说明它负责哪些因素、输出哪些因子。
* 因子 `id` 用稳定的英文下划线命名（会与 `core/config.py` 的权重表匹配），
  例如 `ma_alignment`、`macd_cross`、`capital_main_net`、`valuation_percentile`。
