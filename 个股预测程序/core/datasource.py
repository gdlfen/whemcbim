# -*- coding: utf-8 -*-
"""
多源容错取数模块（Data Source Layer）—— 全项目唯一的数据入口。

职责
----
为"个股涨跌概率预测"提供**统一、可追溯、可降级**的行情数据：

* K 线（日/周/月，前复权/后复权/不复权）→ :func:`fetch_daily` / :func:`fetch_weekly`
  / :func:`fetch_monthly` / :func:`fetch_kline`
* 实时快照（价/量/额/换手/PE/PB/市值/涨跌停/五档盘口）→ :func:`fetch_realtime`
* 指数日线与指数快照 → :func:`fetch_index` / :func:`fetch_index_realtime`
* A 股全市场代码表、代码↔名称互查 → :func:`fetch_stock_list` /
  :func:`resolve_symbol` / :func:`search_symbol`
* 所属板块与板块成分 → :func:`fetch_sector_list` / :func:`fetch_sector_members`
* 数据源健康度体检 → :func:`data_health_check`
* 离线合成数据（演示/自测） → :func:`fetch_daily_synthetic`

铁律
----
1. **绝不抛异常打断上层**：任何联网函数失败都返回空 DataFrame / 空 dict，
   并把失败摘要写入 ``df.attrs["error"]`` 或日志。
2. **绝不返回合成数据**（除非显式调用 :func:`fetch_daily_synthetic`）。
3. **绝不臆造字段**：拿不到就 NaN / None，并用 ``df.attrs`` 标注估算项。
4. **可追溯**：每份返回数据都在 ``df.attrs`` 里写明 ``symbol / source / adjust /
   fetched_at / rows``。

================================ 实测结论（本机 2026-10 实测） ================================

统一输出列（所有 K 线源）
    ``["date","open","high","low","close","volume","amount","pct_chg","turnover"]``
    * ``volume`` 单位统一为 **股**；
    * ``amount`` 单位 **元**，拿不到时按 ``volume*(high+low+close)/3`` 估算并置
      ``attrs["amount_is_estimated"]=True``；
    * ``pct_chg`` 单位 **%**；源未提供时由收盘价一阶差分推算，置
      ``attrs["pct_chg_is_estimated"]=True``（复权序列的差分即真实收益率）；
    * ``turnover`` 单位 **%**；源未提供时置 NaN（换手率需要历史流通股本，
      用当前股本反推会失真，故不臆造）。

1) 腾讯 K 线 ``https://web.ifzq.gtimg.cn/appstock/app/fqkline/get``
   * ``param = <code>,<day|week|month>,<start>,<end>,<count>,<qfq|hfq|空>``（逗号分隔 6 段）
   * **每行字段顺序实测为 ``[date, open, close, high, low, volume]``（O/C/H/L/V，非 OHLC！）**
     用 ``sh600519`` 末行 ``['2026-10-09','1255.420','1263.000','1282.000','1255.420','35111']``
      与 ``qt`` 快照 ``今开=1255.42 现价=1263.00 最高=1282.00 最低=1255.42 量=35111`` 逐项对上确认。
   * **没有第 7 列成交额**。字段数为 6；仅在**除权除息日**多出第 7 列，且它是
     **分红信息 dict**（``{nd, fh_sh, djr, cqr, FHcontent}``，如 ``{'nd':'2023',...,
     'FHcontent':'10派308.76元'}``），**不是成交额**，必须跳过。
     指数（sh000001）恒为 6 列。
   * ``volume`` 单位 = **手**（A 股/指数），需 ×100 换成股。实测 sh600519 末行
     ``35111`` ↔ 新浪同一天 ``3511051`` 股；指数 ``535149921`` 手 ↔ 新浪
     ``53514992100`` 股。**港股/美股返回的已是股**（hk00700 ``20422500``、
     usAAPL ``37881199``），不乘 100。
   * ``count``（第 5 段）**实测上限 800**：``N=800`` → 800 行；``801<=N<=2000``
     → 静默降为 640 行；``N>=2500`` → ``{"code":0,"msg":"param error","data":[]}``。
     因此长历史必须用**第 4 段 end 日期反向分页**（每页 800 根，逐页向前，
     实测 4 页取到 3197 根日线、6 页取到 4795 根）。
   * 第 3/4 段为起止日期（含端点）；两者同时给出且区间内根数多于 count 时，
     **只返回区间内最近的 count 根**。count 段必填（留空 → ``msg="bad params"``）。
   * 周期键名：``qfqday`` / ``qfqweek`` / ``qfqmonth`` / ``hfqday`` / ``hfqweek`` /
     ``hfqmonth`` / 不复权为 ``day`` / ``week`` / ``month``；**指数忽略复权参数，
     键恒为 day/week/month**。
   * 不存在的代码（``sh999999``）：``{"code":0,"msg":"","data":{"sh999999":{"day":[],
     "qt":{"sh999999":[]},...}}}`` → 对应键为空数组。非法周期 → ``code:1,
     msg:"bad params"``，data 内无 K 线键。故必须"扫描子键"而不是硬拼键名。
   * **腾讯 qfq 是"减法前复权"**：早期历史会算出**负价**（实测 sh600519 周线 640 根
     中 115 根负收盘、月线 302 根中 178 根负；2006 年起的日线 4795 根中 2307 根负）。
     本模块的做法：正常区间直接用腾讯 qfq；一旦出现非正价格（**必须在 `_sanitize`
     之前判定**，否则负价行已被丢掉），则改用 ``hfq`` 并按下式重算**乘法前复权**
     （与通达信/同花顺口径一致）::

         前复权_t = 后复权_t × k,   k = 最新不复权收盘 / 最新后复权收盘

     实测支撑：复权因子 ``F = hfq/raw`` 与请求窗口**无关**（分页重叠行数值完全一致），
     且日/周/月线**共用同一条 F 曲线**（137 个周线日期、32/33 个月末日期 F 完全相同）；
     唯一例外是 ``hfqmonth`` 的**最后一根**（当月未走完）用了陈旧因子
     （实测 2026-10-09 F_month=6.3941 而 F_day=7.0556），故 k 统一取**日线锚点**，
     并在检测到"本周期末根 F 偏离日线 F > 0.2%"时用该周期的不复权末根覆盖。
     置 ``attrs["adjust_repaired"]=True`` 与 ``attrs["repair_note"]``；
     与腾讯自带 qfq 在近月差异 0%、越往前差异越大（减法 vs 乘法的口径差异）。

2) 新浪 K 线 ``https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData``
   * ``scale``：``5/15/30/60`` = 分钟；``240`` = 日线；``1200`` = 周线；``7200`` = 月线；
     ``120/2400/4800`` 返回字面量 ``null``（不支持）。
   * ``datalen`` 实测最大约 **6017** 根（6000 → 6000 根；8000/10000 → 6017 根，
     sh600519 可回溯到 2001-08-27）。
   * 字段 ``{"day","open","high","low","close","volume"}``，**顺序为标准 OHLC**；
     ``volume`` 单位 = **股**（与腾讯的"手"不同，这是最容易踩的坑）。
   * **全部为不复权数据**（周线 scale=1200 / 月线 scale=7200 同样不复权），
     返回时 ``attrs["adjust"]="none"`` 并在 ``attrs["note"]`` 里说明。
   * 不存在的代码返回字面量 ``null``（不是 ``[]``），必须防呆。

3) 腾讯实时快照 ``https://qt.gtimg.cn/q=sh600519``（**GBK**，``~`` 分隔，
   响应用 ``v_xxx="..."`` 包裹；未知代码返回 ``v_pv_none_match="1";``）
   实测 A 股 88 字段，逐字段核对后的映射见 :data:`TENCENT_FIELD_MAP`（含依据）。
   注意 **港股 78 字段 / 美股 73 字段，尾部含义完全不同**（如 A 股 [47]/[48] 是
   涨停/跌停价，港股 [48]/[49] 是 52 周高低），故不能用 A 股映射硬套，
   港美股只取前 6 个公共字段 + 单独映射，见 :data:`TENCENT_FOREIGN_FIELD_MAP`。

4) 新浪实时 ``https://hq.sinajs.cn/list=sh600519``（**GBK**，需
   ``Referer: https://finance.sina.com.cn``）实测 34 字段，**买卖五档是"量在前价在后"**
   （与腾讯"价在前量在后"相反），映射见 :data:`SINA_REALTIME_FIELD_MAP`。

5) 指数日线：``akshare.stock_zh_index_daily``（实测 sh000001 = 8738 根，
   1990-12-19 起，date 为 ``datetime.date``，volume 单位 = 股）与新浪指数 K 线
   （同源，volume 单位同为股）。

6) A 股代码表：``akshare.stock_info_a_code_name()`` = 5573 条（沪主板 1702 /
   科创板 617 / 深主板 1495 / 创业板 1409 / 北交所 349，北交所新代码为 ``92xxxx``）。
   东财 clist 全市场接口是备选（本机网络不可达）。

7) yfinance：**本机不可用**，两层原因：
   (a) 时区缓存目录 ``%LOCALAPPDATA%\\py-yfinance\\py-yfinance\\Cache`` 不可写
       → ``OperationalError('unable to open database file')``（本模块会主动把
       缓存目录重定向到可写路径来消除该故障）；
   (b) Yahoo ``query1/query2.finance.yahoo.com`` 返回 **HTTP 403**，
       yfinance 报 ``YFRateLimitError``。
   因此 yfinance 仅作为最后兜底，失败即返回空。

8) 东财 ``push2/push2his`` 行情接口**在本机网络被重置**（实测 6/6 全失败，
   ``RemoteDisconnected``；编号子域 1./7./48./82.、http/https 均无效）：
   ``/api/qt/clist/get``（板块成分/全市场列表）、``/api/qt/stock/kline/get``
   （东财日线）、``/api/qt/stock/fflow/*`` 全部不可用；
   而 **``/api/qt/slist/get``（所属板块）与 ``datacenter-web`` 正常**。
   代码中仍按官方语义实现东财源，运行到其他网络时会自动生效。

9) 数据源体检 :func:`data_health_check`
   * 并发（守护线程 + 总预算 ``timeout``，默认 20s）执行 14 项解析级检查；
   * 另用 ``utils.http_get_many`` **并行**发起 7 个端点的 HTTP 层原始探测
     （``raw_*`` 项），从而区分"网络不可达"与"可达但解析失败"；
   * 代码表整表要 18 页（冷启动 6~19s），因此该项不冷启动：已缓存则秒回，
     未缓存则转后台预热，下次调用即命中 24h 缓存；
   * 实测本机 9~10s 返回，``overall="healthy"``（10~14 项 ok）。

缓存
----
``core.utils.cached/cache_key/cache_get/cache_put`` 两级缓存（内存 + 磁盘 pkl）。
* K 线 TTL = ``config.RUNTIME["cache_ttl_seconds"]``（默认 6h）
* 代码表 TTL = ``config.RUNTIME["cache_ttl_daily"]``（默认 24h）
* 实时快照 TTL = ``config.RUNTIME["cache_ttl_realtime"]``（默认 60s；6 小时前的
  "实时价"没有意义，故此处按数据类型单独设定，可用 ``force=True`` 强制刷新）
* ``force=True`` 跳过读缓存、仍然写缓存；**失败结果（空数据）永不写缓存**。
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import threading
import time
import unicodedata
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np
import pandas as pd

try:  # 包内导入（python -m core.datasource / from core import datasource）
    from . import config
    from .contracts import now_cn, safe_float
    from .utils import (cache_clear, cache_get, cache_key, cache_put, get_logger,
                        http_get, http_get_many, normalize_symbol, retry_call)
except ImportError:  # pragma: no cover - 兼容 python core/datasource.py 直接运行
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core import config
    from core.contracts import now_cn, safe_float
    from core.utils import (cache_clear, cache_get, cache_key, cache_put, get_logger,
                            http_get, http_get_many, normalize_symbol, retry_call)

log = get_logger("stockai.datasource")

__all__ = [
    "KLINE_COLUMNS", "TENCENT_FIELD_MAP", "TENCENT_FOREIGN_FIELD_MAP",
    "SINA_REALTIME_FIELD_MAP",
    "fetch_daily", "fetch_weekly", "fetch_monthly", "fetch_kline",
    "fetch_daily_synthetic", "fetch_realtime", "fetch_index",
    "fetch_index_realtime", "fetch_stock_list", "resolve_symbol", "search_symbol",
    "fetch_sector_list", "fetch_sector_members", "clear_cache", "data_health_check",
]

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
#: 所有 K 线函数的统一输出列
KLINE_COLUMNS = ["date", "open", "high", "low", "close", "volume",
                 "amount", "pct_chg", "turnover"]

_KLINE_DTYPES = {"date": "datetime64[ns]", "open": "float64", "high": "float64",
                 "low": "float64", "close": "float64", "volume": "float64",
                 "amount": "float64", "pct_chg": "float64", "turnover": "float64"}

#: 需要"手 → 股"（×100）的市场前缀；港股/美股腾讯返回的已是股
_LOT_TO_SHARE_PREFIX = ("sh", "sz", "bj")

#: 源返回根数达到请求根数的该比例即认为"够用"，可停止降级
ROWS_SUFFICIENT_RATIO = 0.6

#: 缓存 TTL（秒）
TTL_KLINE = int(config.RUNTIME.get("cache_ttl_seconds", 6 * 3600))     # 行情类
TTL_LIST = int(config.RUNTIME.get("cache_ttl_daily", 24 * 3600))      # 代码表/低频
TTL_REALTIME = int(config.RUNTIME.get("cache_ttl_realtime", 60))      # 实时快照

# ---- 腾讯 K 线 ------------------------------------------------------------ #
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
#: 实测单页 count 上限 = 800（801~2000 会被静默截断为 640，>=2500 直接 param error）
TENCENT_KLINE_MAX_COUNT = 800
TENCENT_KLINE_MAX_PAGES = 12
#: 实测每行顺序 = [date, open, close, high, low, volume]（注意 O/C/H/L/V！）
TENCENT_KLINE_COLUMNS = ["date", "open", "close", "high", "low", "volume"]
TENCENT_KLINE_MIN_DATE = "1990-01-01"
#: 内部周期名 → 腾讯 param 第 2 段的写法（实测只认 day/week/month，
#: 传 "daily" 会返回 {"code":1,"msg":"bad params"}）
TENCENT_PERIOD = {"daily": "day", "weekly": "week", "monthly": "month"}
#: 上面那行顺序的逆映射：列名 → 原始行下标（解析时按名字取，避免硬编码下标出错）
_TX_COL_IDX = {name: i for i, name in enumerate(TENCENT_KLINE_COLUMNS)}


def TENCENT_FIELD_MAP_KEY(field: str) -> int:
    """按字段名反查腾讯实时快照的索引（调试/自测用）。"""
    for idx, (name, _desc) in TENCENT_FIELD_MAP.items():
        if name == field:
            return idx
    raise KeyError(field)

# ---- 新浪 K 线 ------------------------------------------------------------ #
SINA_KLINE_URL = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                  "CN_MarketData.getKLineData")
#: 实测：240=日线、1200=周线、7200=月线（120/2400/4800 返回 null）
SINA_SCALE = {"daily": 240, "weekly": 1200, "monthly": 7200}
#: 实测 datalen 上限约 6017 根
SINA_MAX_LEN = 6017

# ---- 实时快照 ------------------------------------------------------------- #
TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q="
SINA_QUOTE_URL = "https://hq.sinajs.cn/list="
_SINA_HEADERS = {"Referer": "https://finance.sina.com.cn"}

# ---- 东财 ----------------------------------------------------------------- #
EM_SLIST_URL = "https://push2.eastmoney.com/api/qt/slist/get"
EM_CLIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
EM_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
#: 东财 kline 的 fields2 顺序（官方语义；本机 push2his 不可达，未能实测确认）
EM_KLINE_COLUMNS = ["date", "open", "close", "high", "low", "volume", "amount",
                    "amplitude", "pct_chg", "change", "turnover"]
EM_KLINE_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
EM_CLIST_FIELDS = "f12,f14,f2,f3,f9,f20,f23"

#: 常见指数别名（未带市场前缀时使用；避免 000300 被 A 股规则误判为深市个股）
INDEX_ALIASES = {
    "000001": "sh000001", "000016": "sh000016", "000300": "sh000300",
    "000688": "sh000688", "000905": "sh000905", "000852": "sh000852",
    "399001": "sz399001", "399005": "sz399005", "399006": "sz399006",
    "399300": "sz399300", "899050": "bj899050",
}

#: 板块类型映射（东财 clist 的 m:90+t:N；本机不可达时退化为名称启发式）
EM_BOARD_TYPE_FS = {"1": "地域", "2": "行业", "3": "概念"}
#: 板块类型未知时的占位（明确表示"未能确定"，不猜）
BOARD_TYPE_UNKNOWN = "板块"

# --------------------------------------------------------------------------- #
# 腾讯实时快照字段映射（A 股，实测 88 字段）
#
# 依据（2026-10-09 收盘后实测 v_sh600519，现价 1263.00 / 昨收 1255.79 / 量 35111 手）：
#   [03]=1263.00=现价；[04]=1255.79=昨收；[05]=1255.42=今开；[06]=35111=量(手)；
#   [07]+[08]=19932+15179=35111=量 → 外盘/内盘；
#   [09..18]=买一~买五"价,量"交替；[19..28]=卖一~卖五"价,量"交替；
#   [31]=7.21=1263.00-1255.79=涨跌额；[32]=0.57=涨跌幅%；
#   [35]="1263.00/35111/4453223189"=价格/成交量(手)/成交额(元)；
#   [37]=445322=成交额(万元)=4453223189 元；
#   [38]=0.28=换手率%＝3511051 股 / 1250081601 流通股本；
#   [39]=19.39=PE(TTM)；[43]=2.12=(1282.00-1255.42)/1255.79=振幅%；
#   [44]=[45]=15788.53=市值(亿)＝1263.00 × 1250081601 股；
#   [46]=6.28=PB；[47]=1381.37=1255.79×1.1=涨停价；[48]=1130.21=×0.9=跌停价；
#   [49]=1.18=量比；[50]=78=(73+2+3+1+8)-(1+3+1+1+3)=委差(手)；
#   [51]=1268.34=成交额/成交量=均价；[52]=17.73=PE(动)；[53]=19.18=PE(静)；
#   [72]=1250081601=流通股本（换手率分母）；[73]=1250081601=总股本；
#   [82]='CNY'=币种。
# 未识别的索引统一放入 ``extra["fNN"]``，不做臆测命名。
# --------------------------------------------------------------------------- #
TENCENT_FIELD_MAP: dict[int, tuple[str, str]] = {
    0: ("market_flag", "市场标志：1=沪 51=深 62=北 100=港 200=美(delay)"),
    1: ("name", "名称"),
    2: ("code", "代码"),
    3: ("price", "现价"),
    4: ("prev_close", "昨收"),
    5: ("open", "今开"),
    6: ("volume_lots", "成交量(手)"),
    7: ("outer_volume_lots", "外盘(手)"),
    8: ("inner_volume_lots", "内盘(手)"),
    9: ("bid1", "买一价"), 10: ("bid1_lots", "买一量(手)"),
    11: ("bid2", "买二价"), 12: ("bid2_lots", "买二量(手)"),
    13: ("bid3", "买三价"), 14: ("bid3_lots", "买三量(手)"),
    15: ("bid4", "买四价"), 16: ("bid4_lots", "买四量(手)"),
    17: ("bid5", "买五价"), 18: ("bid5_lots", "买五量(手)"),
    19: ("ask1", "卖一价"), 20: ("ask1_lots", "卖一量(手)"),
    21: ("ask2", "卖二价"), 22: ("ask2_lots", "卖二量(手)"),
    23: ("ask3", "卖三价"), 24: ("ask3_lots", "卖三量(手)"),
    25: ("ask4", "卖四价"), 26: ("ask4_lots", "卖四量(手)"),
    27: ("ask5", "卖五价"), 28: ("ask5_lots", "卖五量(手)"),
    29: ("last_tick", "最近逐笔成交（可能为空串）"),
    30: ("time", "时间 YYYYMMDDHHMMSS"),
    31: ("change", "涨跌额"),
    32: ("pct_chg", "涨跌幅%"),
    33: ("high", "最高"),
    34: ("low", "最低"),
    35: ("pva", "价格/成交量(手)/成交额(元)"),
    36: ("volume_lots_dup", "成交量(手) 重复字段"),
    37: ("amount_wan", "成交额(万元)"),
    38: ("turnover", "换手率%"),
    39: ("pe_ttm", "市盈率(TTM)"),
    41: ("high_dup", "最高 重复字段"),
    42: ("low_dup", "最低 重复字段"),
    43: ("amplitude", "振幅%"),
    44: ("float_market_cap_yi", "流通市值(亿)"),
    45: ("market_cap_yi", "总市值(亿)"),
    46: ("pb", "市净率"),
    47: ("limit_up", "涨停价（指数为 -1）"),
    48: ("limit_down", "跌停价（指数为 -1）"),
    49: ("volume_ratio", "量比"),
    50: ("bid_ask_diff_lots", "委差(手) = 五档买量-五档卖量"),
    51: ("avg_price", "均价"),
    52: ("pe_dynamic", "市盈率(动)"),
    53: ("pe_static", "市盈率(静)"),
    57: ("amount_wan_precise", "成交额(万元, 含小数)"),
    60: ("market_tag", "市场标识（如 '   A'）"),
    61: ("security_type", "证券类型（GP-A=股票-A股, ZS=指数）"),
    72: ("float_shares", "流通股本(股)"),
    73: ("total_shares", "总股本(股)"),
    82: ("currency", "币种"),
    84: ("flags", "标记位（如 ___D__F__N）"),
}

#: 港股(78 字段)/美股(73 字段)的**差异映射**：只有这些索引与 A 股同义，
#: 其余一律进 extra。注意港美股 [47]/[48]/[49] 不是涨跌停价。
TENCENT_FOREIGN_FIELD_MAP: dict[int, tuple[str, str]] = {
    3: ("price", "现价"), 4: ("prev_close", "昨收"), 5: ("open", "今开"),
    6: ("volume", "成交量(股)"),
    30: ("time", "时间"), 31: ("change", "涨跌额"), 32: ("pct_chg", "涨跌幅%"),
    33: ("high", "最高"), 34: ("low", "最低"),
    39: ("pe_ttm", "市盈率(或等价估值字段)"),
    43: ("amplitude", "振幅%"),
    44: ("float_market_cap_yi", "流通市值(亿, 本币)"),
    45: ("market_cap_yi", "总市值(亿, 本币)"),
}
#: 港美股成交额/(港)总股本所在索引（实测 hk00700 [37]=成交额(本币)、[36]=成交量）
TENCENT_FOREIGN_AMOUNT_IDX = 37
TENCENT_FOREIGN_SHARES_IDX = {"hk": (70, 69), "us": (63, 62)}   # (流通股本, 总股本)

# --------------------------------------------------------------------------- #
# 新浪实时字段映射（实测 34 字段，"量在前、价在后"）
# 依据（2026-10-09 实测 sh600519：7533/600519 同日腾讯对照）：
#   [00]名称 [01]今开1255.420 [02]昨收1255.790 [03]现价1263.000 [04]最高1282.000
#   [05]最低1255.420 [06]竞买价(买一)1263.000 [07]竞卖价(卖一)1263.060
#   [08]成交量 3511051 **股**（= 腾讯 35111 手 ×100）[09]成交额 4453223189 **元**
#   [10]买一量7344 [11]买一价1263.000 [12]买二量200 [13]买二价1262.990 …
#   [18]买五量 [19]买五价 [20]卖一量 [21]卖一价 … [28]卖五量 [29]卖五价
#   [30]日期 [31]时间 [32]状态 '00' [33]盘后固定价格交易 '量|金额'
# --------------------------------------------------------------------------- #
SINA_REALTIME_FIELD_MAP: dict[int, tuple[str, str]] = {
    0: ("name", "名称"), 1: ("open", "今开"), 2: ("prev_close", "昨收"),
    3: ("price", "现价"), 4: ("high", "最高"), 5: ("low", "最低"),
    6: ("bid1", "竞买价(买一)"), 7: ("ask1", "竞卖价(卖一)"),
    8: ("volume", "成交量(股)"), 9: ("amount", "成交额(元)"),
    10: ("bid1_vol", "买一量(股)"), 11: ("bid1_px", "买一价"),
    12: ("bid2_vol", "买二量(股)"), 13: ("bid2_px", "买二价"),
    14: ("bid3_vol", "买三量(股)"), 15: ("bid3_px", "买三价"),
    16: ("bid4_vol", "买四量(股)"), 17: ("bid4_px", "买四价"),
    18: ("bid5_vol", "买五量(股)"), 19: ("bid5_px", "买五价"),
    20: ("ask1_vol", "卖一量(股)"), 21: ("ask1_px", "卖一价"),
    22: ("ask2_vol", "卖二量(股)"), 23: ("ask2_px", "卖二价"),
    24: ("ask3_vol", "卖三量(股)"), 25: ("ask3_px", "卖三价"),
    26: ("ask4_vol", "卖四量(股)"), 27: ("ask4_px", "卖四价"),
    28: ("ask5_vol", "卖五量(股)"), 29: ("ask5_px", "卖五价"),
    30: ("date", "日期"), 31: ("time", "时间"), 32: ("status", "状态"),
    33: ("after_hours", "盘后固定价格交易 量|金额"),
}


# --------------------------------------------------------------------------- #
# 内部小工具
# --------------------------------------------------------------------------- #
def _offline() -> bool:
    """是否处于离线模式（离线时所有源函数直接返回空）。"""
    return bool(config.RUNTIME.get("offline", False))


_QUIET_LOCK = threading.RLock()


@contextmanager
def _quiet():
    """临时静音第三方库（akshare 的 tqdm 进度条等）往 **stderr** 的输出。

    为什么只换 ``sys.stderr`` 而不换 ``sys.stdout``：本模块会在
    :func:`data_health_check` 里用多个**守护线程**并发取数，而
    ``contextlib.redirect_stdout`` 是进程级全局替换且非线程安全——工作线程一旦
    把 ``sys.stdout`` 换成 StringIO，主线程随后的 ``print`` 就会写进那个
    临时缓冲区并**永久丢失**（实测导致验收脚本输出被截断）。
    日志用的 ``logging.StreamHandler`` 在导入时就已绑定原始 stderr，因此换掉
    ``sys.stderr`` 不会影响本模块日志，也不会影响主线程输出。
    """
    with _QUIET_LOCK:
        saved = sys.stderr
        sys.stderr = io.StringIO()
        try:
            yield
        finally:
            sys.stderr = saved


def _now_str() -> str:
    return now_cn().strftime("%Y-%m-%d %H:%M:%S")


def _empty_kline(attrs: dict | None = None) -> pd.DataFrame:
    """返回列齐全的空 K 线表（绝不返回 None，避免上层 AttributeError）。"""
    df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in _KLINE_DTYPES.items()},
                      columns=KLINE_COLUMNS)
    if attrs:
        df.attrs.update(attrs)
    return df


def _empty_df(columns: list[str], attrs: dict | None = None) -> pd.DataFrame:
    df = pd.DataFrame(columns=columns)
    if attrs:
        df.attrs.update(attrs)
    return df


def _clone(df: pd.DataFrame) -> pd.DataFrame:
    """复制 DataFrame 并保留 attrs（缓存命中时返回副本，避免上层改坏缓存）。"""
    out = df.copy()
    out.attrs = dict(getattr(df, "attrs", {}) or {})
    return out


def _tx_code(info: dict) -> str:
    """腾讯接口用代码：sh600519 / sz000001 / bj920982 / hk00700 / usAAPL。"""
    p = info.get("prefix", "")
    code = str(info.get("code", "") or "")
    if p in ("sh", "sz", "bj"):
        return p + code
    if p == "hk":
        return "hk" + code.zfill(5)
    return "us" + code.upper()


def _sina_code(info: dict) -> str:
    """新浪接口用代码（A 股/北交所）：sh600519 / sz000001 / bj920982。"""
    return "%s%s" % (info.get("prefix", ""), info.get("code", ""))


def _norm_period(period: str) -> str:
    p = str(period or "daily").strip().lower()
    return {"day": "daily", "d": "daily", "1d": "daily", "daily": "daily",
            "week": "weekly", "w": "weekly", "1w": "weekly", "weekly": "weekly",
            "month": "monthly", "m": "monthly", "1mo": "monthly",
            "monthly": "monthly"}.get(p, "daily")


def _norm_adjust(adjust: str) -> str:
    a = str(adjust or "").strip().lower()
    if a in ("qfq", "hfq"):
        return a
    return "none"


def _ak_adjust(adjust: str) -> str:
    return {"qfq": "qfq", "hfq": "hfq", "none": ""}.get(adjust, "")


def _default_days(period: str) -> int:
    """``days=None`` 时的默认根数（日线取 config.RUNTIME['history_days']）。"""
    if period == "daily":
        return int(config.RUNTIME.get("history_days", 3200) or 3200)
    if period == "weekly":
        return 800          # 约 15 年周线
    return 400              # 约 33 年月线


def _cache_read(key: str, ttl: int, force: bool) -> Any:
    """读缓存；``force=True`` 时短路（跳读但后续仍写）。"""
    if force:
        return None
    return cache_get(key, ttl)


def _cache_write_if_ok(key: str, value: Any, ok: bool) -> None:
    """只缓存**成功**结果；空结果不写缓存，避免把一次网络抖动固化 6 小时。"""
    if ok and value is not None:
        cache_put(key, value)


def _df_of(rows: list[dict], attrs: dict) -> pd.DataFrame:
    """行列表 → 统一列 K 线表（不含清洗，清洗统一交给 :func:`_sanitize`）。"""
    if not rows:
        return _empty_kline(attrs)
    df = pd.DataFrame(rows)
    for c in KLINE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[KLINE_COLUMNS]
    for c, t in _KLINE_DTYPES.items():
        try:
            df[c] = df[c].astype(t)
        except (TypeError, ValueError):
            df[c] = pd.to_numeric(df[c], errors="coerce") if t == "float64" else \
                pd.to_datetime(df[c], errors="coerce")
    df.attrs.update(attrs)
    return df


def _tencent_rows_to_df(rows: list[list], tx_code: str) -> pd.DataFrame:
    """腾讯 K 线原始行 → 统一列。

    每行实测为 ``[date, open, close, high, low, volume]``（O/C/H/L/V，见
    :data:`TENCENT_KLINE_COLUMNS`）；除权除息日的行会多出第 7 个元素，
    且它是**分红信息 dict 而非成交额**（实测 ``{'nd':'2023','fh_sh':'308.76',
    'djr':...,'cqr':...,'FHcontent':'10派308.76元'}``），这里直接丢弃。
    """
    ix = _TX_COL_IDX
    recs = []
    for r in rows:
        if not isinstance(r, (list, tuple)) or len(r) < 6:
            continue
        recs.append({
            "date": r[ix["date"]],
            "open": safe_float(r[ix["open"]]),
            "close": safe_float(r[ix["close"]]),
            "high": safe_float(r[ix["high"]]),
            "low": safe_float(r[ix["low"]]),
            "volume": safe_float(r[ix["volume"]]),
            "amount": np.nan,
            "pct_chg": np.nan,
            "turnover": np.nan,
        })
    return _df_of(recs, {})


# --------------------------------------------------------------------------- #
# 数据质量校验
# --------------------------------------------------------------------------- #
def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """剔除坏行并返回清洗后的副本（被剔除条数写入 ``attrs["dropped"]``）。

    剔除规则（硬性约定）::

        close <= 0                                   —— 含腾讯"减法前复权"的负价
        high  < low
        high  < max(open, close) - 1e-6
        low   > min(open, close) + 1e-6

    额外剔除：关键价格/日期缺失（NaN）、volume < 0、date 无法解析。
    不做任何填充、平滑或插值——坏数据宁可丢弃也不修补。

    Returns
    -------
    pandas.DataFrame
        清洗后的新表；``attrs["dropped"]`` 为剔除总条数，
        ``attrs["dropped_detail"]`` 为按原因分项计数。
    """
    if df is None or len(df) == 0:
        out = _empty_kline(dict(getattr(df, "attrs", {}) or {}) if df is not None else None)
        out.attrs["dropped"] = 0
        out.attrs["dropped_detail"] = {}
        return out

    src_attrs = dict(getattr(df, "attrs", {}) or {})
    work = df.copy()
    for c in KLINE_COLUMNS:
        if c not in work.columns:
            work[c] = np.nan
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for c in ("open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover"):
        work[c] = pd.to_numeric(work[c], errors="coerce")

    detail: dict[str, int] = {}

    def _drop(mask: pd.Series, reason: str) -> None:
        n = int(mask.sum())
        if n:
            detail[reason] = detail.get(reason, 0) + n
        work.drop(work.index[mask], inplace=True)

    _drop(work["date"].isna(), "bad_date")
    for c in ("open", "high", "low", "close"):
        _drop(work[c].isna(), "missing_%s" % c)

    # ---- 四条硬性规则（逐条统计，便于追溯源的数据质量） ---- #
    _drop(work["close"] <= 0, "close<=0")
    _drop(work["high"] < work["low"], "high<low")
    _drop(work["high"] < work[["open", "close"]].max(axis=1) - 1e-6, "high<max(o,c)")
    _drop(work["low"] > work[["open", "close"]].min(axis=1) + 1e-6, "low>min(o,c)")
    _drop(work["volume"].notna() & (work["volume"] < 0), "volume<0")

    work["volume"] = work["volume"].fillna(0.0)
    work.sort_values("date", inplace=True, kind="stable")
    work.reset_index(drop=True, inplace=True)

    out = work[KLINE_COLUMNS]
    out.attrs.update(src_attrs)
    out.attrs["dropped"] = int(sum(detail.values()))
    out.attrs["dropped_detail"] = detail
    return out


def _finalize_kline(df: pd.DataFrame, info: dict, period: str, adjust: str,
                    days: int | None, source: str, **extra_attrs) -> pd.DataFrame:
    """统一收尾：排序、去重、补 pct_chg/amount、裁剪根数、写 attrs。"""
    attrs = dict(getattr(df, "attrs", {}) or {})
    attrs.update(extra_attrs)
    work = df.copy()
    work.attrs = attrs

    if len(work):
        # 去重（同一天保留最后一条）
        before = len(work)
        work = work.sort_values("date", kind="stable")
        work = work.drop_duplicates(subset=["date"], keep="last")
        dup = before - len(work)
        work = work.reset_index(drop=True)
        work.attrs = attrs
        if dup:
            attrs["duplicates_removed"] = int(dup)

        # --- 成交额缺失 → 估算并标注 ---
        if work["amount"].isna().all() or float(work["amount"].notna().sum()) == 0:
            work["amount"] = (work["volume"] * (work["high"] + work["low"]
                                                + work["close"]) / 3.0)
            attrs["amount_is_estimated"] = True
        else:
            attrs["amount_is_estimated"] = False

        # --- 涨跌幅缺失 → 由收盘价差分推算（复权序列的差分即真实收益率） ---
        if work["pct_chg"].isna().all() or float(work["pct_chg"].notna().sum()) == 0:
            work["pct_chg"] = work["close"].pct_change() * 100.0
            attrs["pct_chg_is_estimated"] = True
        else:
            attrs["pct_chg_is_estimated"] = False

        # --- 换手率：源不给就保持 NaN（不臆造） ---
        attrs["turnover_available"] = bool(work["turnover"].notna().any())

        if days and len(work) > int(days):
            work = work.iloc[-int(days):].reset_index(drop=True)
            work.attrs = attrs

    attrs.update({
        "symbol": info.get("display") or info.get("code"),
        "code": info.get("code"),
        "name": attrs.get("name"),
        "prefix": info.get("prefix"),
        "period": period,
        "source": source,
        "adjust": adjust,
        "requested_adjust": attrs.get("requested_adjust", adjust),
        "fetched_at": _now_str(),
        "rows": int(len(work)),
    })
    if adjust == "none" and attrs.get("requested_adjust") not in (None, "none"):
        attrs["note"] = ("%s 源为不复权数据，无法提供 %s；返回数据 adjust=none"
                         % (source, attrs["requested_adjust"]))
    work.attrs = attrs
    return work


# --------------------------------------------------------------------------- #
# 源 1：腾讯 K 线
# --------------------------------------------------------------------------- #
#: 腾讯响应内嵌的 qt 快照最近一次解析到的名称（形如 {"sh600519": "贵州茅台"}）
_TX_LAST_NAME: dict[str, str] = {}
#: 腾讯复权锚点缓存：tx_code -> (最新不复权收盘, 最新后复权收盘)
_TX_ANCHOR: dict[str, tuple[float, float]] = {}


def _tencent_fetch_raw(tx_code: str, period: str, end: str, count: int,
                       adjust: str) -> list[list]:
    """抓腾讯 K 线的一页。

    ``param`` 六段语义（实测）：``code,period,start,end,count,adjust``

    * 第 2 段 = ``day`` / ``week`` / ``month``（**实测不认 daily/weekly/monthly**，
      传错直接 ``{"code":1,"msg":"bad params"}``）；
    * 第 3/4 段 = 起止日期（含端点），可留空；两者给出且区间内根数多于 count 时
      **只返回区间内最近的 count 根**；
    * 第 5 段 = 根数，**必填且实测上限 800**（801~2000 静默截断为 640，
      >=2500 直接 ``param error``）；
    * 第 6 段 = 复权方式，可留空表示不复权。
    """
    tx_period = TENCENT_PERIOD.get(period, "day")
    param = "%s,%s,,%s,%d,%s" % (tx_code, tx_period, end or "",
                                 max(1, min(int(count), TENCENT_KLINE_MAX_COUNT)),
                                 adjust if adjust in ("qfq", "hfq") else "")
    txt = http_get(TENCENT_KLINE_URL, params={"param": param}, timeout=15, retries=2)
    data = json.loads(txt) if isinstance(txt, str) else txt
    if not isinstance(data, dict):
        return []
    msg = str(data.get("msg") or "")
    if msg and msg not in ("", "0"):
        raise RuntimeError("tencent kline msg=%s (param=%s)" % (msg, param))
    node = (data.get("data") or {})
    if not isinstance(node, dict):
        return []
    node = node.get(tx_code) or {}
    if not isinstance(node, dict):
        return []

    # K 线响应里内嵌了实时快照 qt，顺手取出名称（腾讯是唯一能免费给名称的 K 线源）
    qt = node.get("qt")
    if isinstance(qt, dict):
        arr = qt.get(tx_code)
        if isinstance(arr, list) and len(arr) > 1 and arr[1]:
            _TX_LAST_NAME[tx_code] = str(arr[1])

    # 键名实测：qfqday/qfqweek/qfqmonth、hfqday/...、day/week/month
    # 指数忽略复权参数（键恒为 day/week/month），无数据时键为空数组，
    # 因此按候选键优先、再扫描任意"列表的列表"兜底。
    for key in ("%s%s" % (adjust, tx_period) if adjust in ("qfq", "hfq") else tx_period,
                tx_period, "qfq" + tx_period, "hfq" + tx_period):
        v = node.get(key)
        if isinstance(v, list) and v and isinstance(v[0], list):
            return v
    best: list[list] = []
    for k, v in node.items():
        if isinstance(v, list) and v and isinstance(v[0], list) and len(v) > len(best):
            best = v
    return best


def _tencent_volume_to_shares(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """A 股/指数：腾讯 volume 单位是**手**，×100 换成股；港美股已是股，不换算。"""
    if prefix in _LOT_TO_SHARE_PREFIX and len(df):
        df["volume"] = df["volume"] * 100.0
    return df


def _daily_tencent(info: dict, period: str, days: int, adjust: str) -> pd.DataFrame:
    """腾讯 K 线源（支持 qfq/hfq，日/周/月），按 end 日期反向分页取长历史。"""
    attrs = {"source": "tencent", "requested_adjust": adjust, "name": None}
    if _offline():
        return _empty_kline(dict(attrs, error="offline 模式跳过联网"))
    tx_code = _tx_code(info)
    prefix = info.get("prefix", "")

    collected: list[list] = []
    end = ""
    pages = 0
    want = int(days) + 5
    while pages < TENCENT_KLINE_MAX_PAGES and (len(collected) < want or pages == 0):
        rows = _tencent_fetch_raw(tx_code, period, end, TENCENT_KLINE_MAX_COUNT, adjust)
        pages += 1
        if not rows:
            break
        first_date = str(rows[0][0])
        # 只保留比上一页更早的部分，避免重复
        fresh = [r for r in rows if not collected or str(r[0]) < end]
        collected = fresh + collected
        if not fresh or first_date <= TENCENT_KLINE_MIN_DATE or len(rows) < TENCENT_KLINE_MAX_COUNT:
            break
        end = first_date
    if not collected:
        return _empty_kline(dict(attrs, error="腾讯 K 线返回空（代码 %s 可能不存在）" % tx_code))

    df = _tencent_rows_to_df(collected, tx_code)
    df = _tencent_volume_to_shares(df, prefix)

    # ---- 减法前复权负价检测：必须在 _sanitize 之前判断（清洗会把负价行丢掉） ----
    need_rescue = bool(adjust == "qfq" and len(df)
                       and (pd.to_numeric(df["close"], errors="coerce") <= 0).any())

    # 长历史会触发 hfq 兜底取数，先把 hfq 一起抓下来（若需要）
    if not need_rescue:
        df = _sanitize(df)
    else:
        df = _tencent_rescue_qfq(info, period, days, df)
    df.attrs.update(attrs)
    df.attrs["pages"] = pages
    df.attrs["name"] = _TX_LAST_NAME.get(tx_code) or df.attrs.get("name")
    if df.attrs.get("dropped"):
        df.attrs["note"] = ("腾讯源剔除 %d 根坏 K 线（%s）"
                            % (df.attrs["dropped"], df.attrs.get("dropped_detail")))
    return df


def _tencent_qfq_anchor(tx_code: str) -> tuple[float, float]:
    """腾讯复权锚点 ``(最新不复权收盘, 最新后复权收盘)``。

    实测结论：腾讯的复权因子 F=hfq/raw 是**与请求窗口无关的全局函数**
    （分页重叠行数值完全一致、不同 count 的末行数值一致），且日/周/月线共用同一条
    F 曲线（2023-06-30、2024-06-28 等 137 个周线日期、32/33 个月末日期 F 完全相同）。
    **唯一例外**：``hfqmonth`` 的最后一根（当月未走完）用的是陈旧因子
    （实测 2026-10-09 F_month=6.394149 而 F_day=7.055570），因此统一用**日线锚点**。
    """
    if tx_code in _TX_ANCHOR:
        return _TX_ANCHOR[tx_code]
    raw_rows = _tencent_fetch_raw(tx_code, "daily", "", 3, "")
    hfq_rows = _tencent_fetch_raw(tx_code, "daily", "", 3, "hfq")
    rl = safe_float(raw_rows[-1][2]) if raw_rows else float("nan")
    hl = safe_float(hfq_rows[-1][2]) if hfq_rows else float("nan")
    _TX_ANCHOR[tx_code] = (rl, hl)
    return rl, hl


def _tencent_rescue_qfq(info: dict, period: str, days: int,
                        df_raw_qfq: pd.DataFrame) -> pd.DataFrame:
    """腾讯 qfq 是"减法前复权"，长历史会算出负价；改用 hfq 重算**乘法前复权**。

    恒等式（前复权锚定最新收盘价）::

        前复权_t = 后复权_t × k,        k = 最新不复权收盘 / 最新后复权收盘
        即  k = raw_T / hfq_T   （等价于 qfq_t = raw_t × F_t / F_T）

    * k 统一取**日线**锚点：实测 F 曲线与周期无关，而 ``hfqmonth`` 的最后一根
      是陈旧因子，不能用作锚点；
    * 若本周期最后一根的 F 与日线 F 明显不符（>0.2%），说明该根是陈旧值，
      直接用该周期的**不复权**最后一根覆盖（前复权的当期值恒等于不复权值）；
    * 重算结果与腾讯自带 qfq 在近月差异为 0，越往前差异越大（腾讯用减法、
      这里用乘法，与通达信/同花顺口径一致），属**口径差异而非错误**。
    """
    tx_code = _tx_code(info)
    attrs = dict(getattr(df_raw_qfq, "attrs", {}) or {})
    bad = int((pd.to_numeric(df_raw_qfq["close"], errors="coerce") <= 0).sum())
    fallback = _sanitize(df_raw_qfq)
    fallback.attrs.update(attrs)
    fallback.attrs["note"] = ("腾讯 qfq 出现 %d 根非正价（减法前复权），"
                             "hfq 兜底不可用，已剔除坏行" % bad)

    def _paged_hfq() -> list[list]:
        rows_all: list[list] = []
        end = ""
        page = 0
        want = int(days) + 5
        while page < TENCENT_KLINE_MAX_PAGES and (len(rows_all) < want or page == 0):
            rows = _tencent_fetch_raw(tx_code, period, end, TENCENT_KLINE_MAX_COUNT, "hfq")
            page += 1
            if not rows:
                break
            first_date = str(rows[0][0])
            fresh = [r for r in rows if not rows_all or str(r[0]) < end]
            rows_all = fresh + rows_all
            if not fresh or first_date <= TENCENT_KLINE_MIN_DATE \
                    or len(rows) < TENCENT_KLINE_MAX_COUNT:
                break
            end = first_date
        return rows_all

    try:
        hfq_rows = _paged_hfq()
        raw_day, hfq_day = _tencent_qfq_anchor(tx_code)
        if not hfq_rows or not (raw_day > 0 and hfq_day > 0):
            return fallback
        k = raw_day / hfq_day

        df = _tencent_rows_to_df(hfq_rows, tx_code)
        df = _tencent_volume_to_shares(df, info.get("prefix", ""))
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] * k

        # ---- 最后一根陈旧因子修正 ----
        raw_last_rows = _tencent_fetch_raw(tx_code, period, "", 3, "")
        fixed_last = False
        if raw_last_rows and len(df):
            last_date = str(hfq_rows[-1][0])
            if str(raw_last_rows[-1][0]) == last_date:
                hfq_last = safe_float(hfq_rows[-1][2])
                raw_last = safe_float(raw_last_rows[-1][2])
                if hfq_last > 0 and raw_last > 0:
                    f_ratio = (hfq_last / raw_last) / (hfq_day / raw_day)
                    if abs(f_ratio - 1.0) > 0.002:      # 本周期末根为陈旧复权因子
                        idx = df.index[-1]
                        for i, c in enumerate(("open", "close", "high", "low")):
                            df.loc[idx, c] = safe_float(raw_last_rows[-1][i + 1])
                        fixed_last = True

        df = _sanitize(df)
        if df.empty:
            return fallback
        log.info("kline 源 tencent 的 qfq 出现 %d 根负价（减法复权），"
                 "已用 hfq × %.6f（日线锚点）重算乘法前复权：%d 根%s",
                 bad, k, len(df), "（末根陈旧因子已用不复权值覆盖）" if fixed_last else "")
        df.attrs.update(attrs)
        df.attrs["adjust_repaired"] = True
        df.attrs["repair_note"] = (
            "腾讯 qfq 为减法前复权，长历史出现 %d 根非正价；已按 "
            "前复权=hfq×(最新不复权收盘/最新后复权收盘) 重算为乘法前复权"
            "（k=%.8f，与通达信/同花顺口径一致）" % (bad, k))
        return df
    except Exception as exc:  # noqa: BLE001
        log.info("kline 源 %s 的 qfq 负价修复失败：%s", "tencent", exc)
        return fallback


# --------------------------------------------------------------------------- #
# 源 2：新浪 K 线（不复权，最长 ~6017 根，volume 单位=股）
# --------------------------------------------------------------------------- #
def _daily_sina(info: dict, period: str, days: int, adjust: str) -> pd.DataFrame:
    """新浪 K 线源：**不复权**，volume 已是股；实测 datalen 上限 ~6017。"""
    attrs = {"source": "sina", "requested_adjust": adjust, "name": None}
    if _offline():
        return _empty_kline(dict(attrs, error="offline 模式跳过联网"))
    sym = _sina_code(info)
    scale = SINA_SCALE.get(period, 240)
    datalen = int(min(max(int(days) + 60, 120), SINA_MAX_LEN))
    try:
        txt = http_get(SINA_KLINE_URL,
                       params={"symbol": sym, "scale": scale, "ma": "no",
                               "datalen": datalen},
                       timeout=20, retries=2)
        # 实测：不存在的代码返回字面量 null（不是 []）
        arr = json.loads(txt) if isinstance(txt, str) else txt
    except Exception as exc:  # noqa: BLE001
        return _empty_kline(dict(attrs, error="新浪 K 线请求失败: %s" % exc))
    if not isinstance(arr, list) or not arr:
        return _empty_kline(dict(attrs, error="新浪 K 线返回空/非数组（%s）" % sym))

    recs = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        recs.append({
            "date": it.get("day") or it.get("date"),
            "open": safe_float(it.get("open")),
            "high": safe_float(it.get("high")),
            "low": safe_float(it.get("low")),
            "close": safe_float(it.get("close")),
            "volume": safe_float(it.get("volume")),      # 单位：股（无需 ×100）
            "amount": np.nan,
            "pct_chg": np.nan,
            "turnover": np.nan,
        })
    df = _sanitize(_df_of(recs, attrs))
    df.attrs.update(attrs)
    # 新浪只提供不复权数据
    df.attrs["adjust"] = "none"
    df.attrs["note"] = "新浪 K 线为不复权数据（scale=%d）" % scale
    return df


# --------------------------------------------------------------------------- #
# 源 3：东财 K 线（支持 fqt=1 前复权 / 2 后复权；本机网络不可达，按官方语义实现）
# --------------------------------------------------------------------------- #
def _daily_eastmoney(info: dict, period: str, days: int, adjust: str) -> pd.DataFrame:
    """东财 K 线源。``fields2`` 顺序：日期,开,收,高,低,量(手),额(元),振幅,涨跌幅,涨跌额,换手率。"""
    attrs = {"source": "eastmoney", "requested_adjust": adjust, "name": None}
    if _offline():
        return _empty_kline(dict(attrs, error="offline 模式跳过联网"))
    klt = {"daily": 101, "weekly": 102, "monthly": 103}.get(period, 101)
    fqt = {"qfq": 1, "hfq": 2, "none": 0}.get(adjust, 1)
    secid = info.get("secid") or "1.600519"

    def _call() -> dict:
        txt = http_get(EM_KLINE_URL, params={
            "secid": secid, "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": EM_KLINE_FIELDS2, "klt": klt, "fqt": fqt,
            "beg": 0, "end": 20500101, "lmt": int(min(max(int(days) + 30, 120), 10000)),
        }, timeout=12, retries=1)
        return json.loads(txt) if isinstance(txt, str) else txt

    data = retry_call(_call, attempts=3, delay=0.6, default=None, label="em_kline")
    if not isinstance(data, dict):
        return _empty_kline(dict(attrs, error="东财 K 线请求失败（push2his 不可达）"))
    node = data.get("data") or {}
    klines = node.get("klines") or []
    if not klines:
        return _empty_kline(dict(attrs, error="东财 K 线返回空（%s）" % secid))

    recs = []
    for line in klines:
        cells = str(line).split(",")
        if len(cells) < 6:
            continue
        rec = {"date": cells[0], "open": safe_float(cells[1]), "close": safe_float(cells[2]),
               "high": safe_float(cells[3]), "low": safe_float(cells[4]),
               "volume": safe_float(cells[5]), "amount": np.nan, "pct_chg": np.nan,
               "turnover": np.nan}
        if len(cells) > 6:
            rec["amount"] = safe_float(cells[6])
        if len(cells) > 8:
            rec["pct_chg"] = safe_float(cells[8])
        if len(cells) > 10:
            rec["turnover"] = safe_float(cells[10])
        recs.append(rec)
    df = _df_of(recs, attrs)
    # 东财 volume 单位为手 → 股
    if len(df):
        df["volume"] = df["volume"] * 100.0
    df = _sanitize(df)
    df.attrs.update(attrs)
    df.attrs["name"] = node.get("name")
    return df


# --------------------------------------------------------------------------- #
# 源 4：akshare（内部走东财，字段语义与东财一致）
# --------------------------------------------------------------------------- #
def _daily_akshare(info: dict, period: str, days: int, adjust: str) -> pd.DataFrame:
    """akshare 日线源（``stock_zh_a_hist``）：成交量单位为**手**，需 ×100。"""
    attrs = {"source": "akshare", "requested_adjust": adjust, "name": None}
    if _offline():
        return _empty_kline(dict(attrs, error="offline 模式跳过联网"))
    if not info.get("is_cn"):
        return _empty_kline(dict(attrs, error="akshare 该接口仅支持 A 股"))
    try:
        import akshare as ak
    except Exception as exc:  # noqa: BLE001
        return _empty_kline(dict(attrs, error="akshare 不可用: %s" % exc))

    period_cn = {"daily": "daily", "weekly": "weekly", "monthly": "monthly"}[period]
    end = now_cn().strftime("%Y%m%d")
    # 按根数估算起始日期（日线 1.6 自然日/交易日，周线 7.6，月线 31）
    span_days = int(days * {"daily": 1.7, "weekly": 7.7, "monthly": 32.0}[period]) + 200
    start = (now_cn() - pd.Timedelta(days=span_days)).strftime("%Y%m%d")
    try:
        with _quiet():
            raw = ak.stock_zh_a_hist(symbol=str(info.get("code")), period=period_cn,
                                     start_date=start, end_date=end,
                                     adjust=_ak_adjust(adjust))
    except Exception as exc:  # noqa: BLE001
        return _empty_kline(dict(attrs, error="akshare stock_zh_a_hist 失败: %s" % exc))
    if raw is None or len(raw) == 0:
        return _empty_kline(dict(attrs, error="akshare 返回空"))

    cols = {str(c): c for c in raw.columns}

    def _col(*names):
        for n in names:
            if n in cols:
                return raw[cols[n]]
        return pd.Series([np.nan] * len(raw), index=raw.index)

    recs = pd.DataFrame({
        "date": _col("日期", "date"),
        "open": _col("开盘", "open"),
        "close": _col("收盘", "close"),
        "high": _col("最高", "high"),
        "low": _col("最低", "low"),
        "volume": _col("成交量", "volume"),      # 手
        "amount": _col("成交额", "amount"),
        "pct_chg": _col("涨跌幅", "pct_chg"),
        "turnover": _col("换手率", "turnover"),
    })
    df = _df_of(recs.to_dict("records"), attrs)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100.0   # 手 → 股
    df = _sanitize(df)
    df.attrs.update(attrs)
    return df


# --------------------------------------------------------------------------- #
# 源 5：yfinance（最后兜底；本机不可用，见模块 docstring）
# --------------------------------------------------------------------------- #
_YF_READY: bool | None = None
_YF_LOCK = threading.Lock()


def _yf_module():
    """惰性导入 yfinance，并把时区缓存重定向到可写目录。

    实测故障：yfinance 默认时区缓存目录（``%LOCALAPPDATA%\\py-yfinance\\...``）
    在受限环境下不可写，会把所有请求变成
    ``OperationalError('unable to open database file')``；重定向后错误消失。
    """
    global _YF_READY
    with _YF_LOCK:
        if _YF_READY is False:
            return None
        try:
            import yfinance as yf
        except Exception as exc:  # noqa: BLE001
            log.info("yfinance 不可用：%s", exc)
            _YF_READY = False
            return None
        if _YF_READY is None:
            try:
                base = config.ensure_cache_dir()
                tz = os.path.join(base, "yf_tz")
                os.makedirs(tz, exist_ok=True)
                yf.set_tz_cache_location(tz)
                _YF_READY = True
            except Exception as exc:  # noqa: BLE001
                log.info("yfinance 时区缓存重定向失败：%s", exc)
                _YF_READY = True   # 库本身仍可用，只是可能报缓存错
            # yfinance 会把限流/下载失败用 ERROR 级别刷屏（本模块已自行降级处理）
            try:
                import logging as _logging
                _logging.getLogger("yfinance").setLevel(_logging.CRITICAL)
            except Exception:  # noqa: BLE001
                pass
        return yf


def _daily_yfinance(info: dict, period: str, days: int, adjust: str) -> pd.DataFrame:
    """yfinance 源：**不复权**（auto_adjust=False），volume 单位=股。"""
    attrs = {"source": "yfinance", "requested_adjust": adjust, "name": None}
    if _offline():
        return _empty_kline(dict(attrs, error="offline 模式跳过联网"))
    yf = _yf_module()
    if yf is None:
        return _empty_kline(dict(attrs, error="yfinance 未安装/不可用"))
    sym = info.get("yf_symbol") or info.get("code")
    interval = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}[period]
    span = int(days * {"daily": 1.7, "weekly": 7.7, "monthly": 32.0}[period]) + 200
    start = (now_cn() - pd.Timedelta(days=min(span, 365 * 25))).strftime("%Y-%m-%d")
    try:
        with _quiet():
            raw = yf.download(sym, start=start, interval=interval, progress=False,
                              auto_adjust=False, threads=False)
    except Exception as exc:  # noqa: BLE001
        return _empty_kline(dict(attrs, error="yfinance 失败: %s" % exc))
    if raw is None or len(raw) == 0:
        return _empty_kline(dict(attrs, error="yfinance 返回空（%s；本机 Yahoo 返回 403）" % sym))

    # 单标的 → 列可能是 MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            raw = raw.xs(sym, axis=1, level=-1)
        except Exception:  # noqa: BLE001
            raw.columns = [c[0] for c in raw.columns]
    raw = raw.reset_index()
    cols = {str(c).lower(): c for c in raw.columns}

    def _col(*names):
        for n in names:
            if n in cols:
                return raw[cols[n]]
        return pd.Series([np.nan] * len(raw), index=raw.index)

    recs = pd.DataFrame({
        "date": _col("date", "datetime", "index"),
        "open": _col("open"),
        "high": _col("high"),
        "low": _col("low"),
        "close": _col("close"),
        "volume": _col("volume"),
    })
    recs["amount"] = np.nan
    recs["pct_chg"] = np.nan
    recs["turnover"] = np.nan
    df = _sanitize(_df_of(recs.to_dict("records"), attrs))
    df.attrs.update(attrs)
    try:  # yfinance 的日期带时区，去掉时区信息便于统一处理
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return df


#: 各 K 线源 → (实现, 是否支持复权)
_KLINE_SOURCES: dict[str, tuple[Callable, bool]] = {
    "sina": (_daily_sina, False),
    "tencent": (_daily_tencent, True),
    "eastmoney": (_daily_eastmoney, True),
    "akshare": (_daily_akshare, True),
    "yfinance": (_daily_yfinance, False),
}

#: 各市场可用的源（A 股专用接口对港美股跳过；腾讯支持港股但美股 K 线残缺）
_SOURCES_BY_MARKET = {
    "cn": ("sina", "tencent", "eastmoney", "akshare", "yfinance"),
    "hk": ("tencent", "yfinance"),
    "us": ("yfinance",),
}


def _kline_multi_source(info: dict, period: str, days: int,
                        adjust: str) -> tuple[pd.DataFrame, str, list[str]]:
    """多源降级取 K 线，返回 ``(df, source, errors)``。

    策略（严格遵守 config.DATASOURCE_PRIORITY 的**相对顺序**）：

    1. 先按 ``DATASOURCE_PRIORITY["kline_<period>"]`` 取顺序，并按市场过滤；
    2. 请求复权时，把**不支持复权的源整体后移**作为兜底——否则会把不复权价
       当复权价返回，在除权日制造假暴跌（这是数据质量事故，不是降级）；
    3. 逐个尝试：任一源根数达到 ``days*60%`` 即采用并停止；
       否则记录根数最多的结果继续尝试，最终返回根数最多者。
    """
    market = info.get("market", "cn")
    if _offline():                     # 离线短路：不打网络、也不刷降级日志
        return _empty_kline({"error": "offline 模式（config.RUNTIME['offline']=True）",
                             "offline": True}), "", ["offline"]
    order = list(config.DATASOURCE_PRIORITY.get(
        "kline_%s" % period, config.DATASOURCE_PRIORITY["kline_daily"]))
    allowed = _SOURCES_BY_MARKET.get(market, ("yfinance",))
    order = [s for s in order if s in allowed and s in _KLINE_SOURCES]
    if adjust != "none":
        order = ([s for s in order if _KLINE_SOURCES[s][1]]
                 + [s for s in order if not _KLINE_SOURCES[s][1]])

    need = max(1, int(days * ROWS_SUFFICIENT_RATIO))
    best: pd.DataFrame | None = None
    best_src = ""
    errors: list[str] = []
    for i, name in enumerate(order):
        fn = _KLINE_SOURCES[name][0]
        try:
            df = fn(info, period, days, adjust)
        except Exception as exc:  # noqa: BLE001 - 任何源异常都必须降级
            df = _empty_kline()
            errors.append("%s: %s: %s" % (name, type(exc).__name__, exc))
        if df is None or len(df) == 0:
            reason = ""
            if df is not None and df.attrs.get("error"):
                reason = str(df.attrs["error"])
            errors.append("%s: %s" % (name, reason or "空结果"))
            nxt = order[i + 1] if i + 1 < len(order) else "(无更多源)"
            log.info("kline 源 %s 失败，降级到 %s", name, nxt)
            continue
        if best is None or len(df) > len(best):
            best, best_src = df, name
        if len(df) >= need:
            break
        nxt = order[i + 1] if i + 1 < len(order) else "(无更多源)"
        log.info("kline 源 %s 仅 %d 根(<%d)，降级到 %s", name, len(df), need, nxt)
    if best is None:
        return _empty_kline(), "", errors
    return best, best_src, errors


# --------------------------------------------------------------------------- #
# 公开 K 线接口
# --------------------------------------------------------------------------- #
def fetch_kline(symbol, period: str = "daily", days: int | None = None,
                adjust: str = "qfq", force: bool = False) -> pd.DataFrame:
    """多源容错 K 线（统一列 / 统一单位 / 带 attrs 溯源）。

    Parameters
    ----------
    symbol : str
        ``600519`` / ``sh600519`` / ``600519.SH`` / ``00700.HK`` / ``AAPL``。
    period : str
        ``daily`` / ``weekly`` / ``monthly``（兼容 ``day/week/month/d/w/m``）。
    days : int | None
        期望返回的**根数**（日线=交易日数，周线=周数，月线=月数）；
        ``None`` 时取 ``config.RUNTIME["history_days"]``（日线 3200 / 周线 800 / 月线 400）。
        实际返回不超过该根数（不足则返回全部）。
    adjust : str
        ``"qfq"`` 前复权（默认）/ ``"hfq"`` 后复权 / ``"none"`` 不复权。
        新浪与 yfinance 只提供不复权，作为兜底源使用时
        ``df.attrs["adjust"]`` 会被改写成 ``"none"`` 并在 ``attrs["note"]`` 说明。
    force : bool
        跳过读缓存（仍写缓存）。

    Returns
    -------
    pandas.DataFrame
        列固定为 :data:`KLINE_COLUMNS`，``date`` 升序去重；失败时返回**列齐全的空表**，
        失败摘要见 ``df.attrs["error"]``。``attrs`` 含
        ``symbol/source/adjust/fetched_at/rows/period/dropped`` 等。
    """
    period = _norm_period(period)
    adjust = _norm_adjust(adjust)
    req_adjust = adjust
    info = normalize_symbol(symbol)
    n_days = int(days) if days else _default_days(period)
    key = cache_key("kline", info.get("display"), period, adjust, n_days)

    hit = _cache_read(key, TTL_KLINE, force)
    if isinstance(hit, pd.DataFrame) and len(hit):
        out = _clone(hit)
        out.attrs["cached"] = True
        return out

    if _offline():                      # 离线短路（不发请求、不刷降级日志）
        return _empty_kline({
            "symbol": info.get("display"), "code": info.get("code"), "period": period,
            "source": None, "adjust": adjust, "requested_adjust": req_adjust,
            "fetched_at": _now_str(), "rows": 0, "cached": False, "offline": True,
            "error": "offline 模式（config.RUNTIME['offline']=True），未发起联网请求",
        })

    df, source, errors = _kline_multi_source(info, period, n_days, adjust)
    if df is None or len(df) == 0:
        out = _empty_kline({
            "symbol": info.get("display"), "code": info.get("code"), "period": period,
            "source": None, "adjust": adjust, "requested_adjust": req_adjust,
            "fetched_at": _now_str(), "rows": 0, "cached": False,
            "error": "所有 K 线源均失败: " + "; ".join(errors[:6]),
        })
        log.info("kline %s %s 全部源失败：%s", info.get("display"), period, errors[:3])
        return out

    out = _finalize_kline(df, info, period, df.attrs.get("adjust", adjust),
                          n_days, source, requested_adjust=req_adjust,
                          cached=False)
    if errors:
        out.attrs["degraded_from"] = errors
    _cache_write_if_ok(key, out, ok=len(out) > 0)
    return out


def fetch_daily(symbol, days: int | None = None, adjust: str = "qfq",
                force: bool = False) -> pd.DataFrame:
    """日线（见 :func:`fetch_kline`；``volume`` 单位为股，``amount`` 单位为元）。"""
    return fetch_kline(symbol, period="daily", days=days, adjust=adjust, force=force)


def fetch_weekly(symbol, days: int | None = None, adjust: str = "qfq",
                 force: bool = False) -> pd.DataFrame:
    """周线（``days`` 为**周数**；新浪 scale=1200 为不复权，腾讯为 qfq/hfq）。"""
    return fetch_kline(symbol, period="weekly", days=days, adjust=adjust, force=force)


def fetch_monthly(symbol, days: int | None = None, adjust: str = "qfq",
                  force: bool = False) -> pd.DataFrame:
    """月线（``days`` 为**月数**；新浪 scale=7200 为不复权，腾讯为 qfq/hfq）。"""
    return fetch_kline(symbol, period="monthly", days=days, adjust=adjust, force=force)


def fetch_daily_synthetic(symbol="600519", days: int = 900, seed: int = 42) -> pd.DataFrame:
    """离线合成日线（演示/自测专用，**明确标注** ``attrs["source"]="synthetic"``）。

    直接调用 :func:`core.selftest_data.make_ohlcv`，结果确定（同 seed 同结果）。
    """
    from .selftest_data import make_ohlcv  # 延迟导入，避免模块级循环依赖

    info = normalize_symbol(symbol)
    n = max(30, int(days))
    try:
        df = make_ohlcv(n=n, seed=int(seed), symbol=str(info.get("code") or symbol))
    except Exception as exc:  # noqa: BLE001
        return _empty_kline({"symbol": info.get("display"), "source": "synthetic",
                             "rows": 0, "error": "合成数据生成失败: %s" % exc})
    for c in KLINE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[KLINE_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df.attrs = {
        "symbol": info.get("display") or str(symbol), "code": info.get("code"),
        "name": None, "period": "daily", "source": "synthetic", "adjust": "qfq",
        "requested_adjust": "qfq", "fetched_at": _now_str(), "rows": int(len(df)),
        "synthetic": True, "seed": int(seed), "amount_is_estimated": False,
        "pct_chg_is_estimated": False, "turnover_available": True, "dropped": 0,
        "note": "合成数据，仅用于离线演示与自测，禁止用于真实投资判断",
    }
    return df


# --------------------------------------------------------------------------- #
# 指数
# --------------------------------------------------------------------------- #
def _norm_index_symbol(symbol) -> dict:
    """指数代码规范化（未带市场前缀时用 :data:`INDEX_ALIASES` 兜底）。"""
    raw = str(symbol or "").strip()
    if raw.upper() in INDEX_ALIASES:
        raw = INDEX_ALIASES[raw.upper()]
    elif re.fullmatch(r"\d{6}", raw) and raw in INDEX_ALIASES:
        raw = INDEX_ALIASES[raw]
    return normalize_symbol(raw)


def _index_sina(info: dict, days: int) -> pd.DataFrame:
    """新浪指数日线（实测 sh000001 可取 6017 根，volume 单位=股）。"""
    attrs = {"source": "sina", "requested_adjust": "none"}
    if _offline():
        return _empty_kline(dict(attrs, error="offline 模式跳过联网"))
    try:
        txt = http_get(SINA_KLINE_URL,
                       params={"symbol": _sina_code(info), "scale": 240, "ma": "no",
                               "datalen": int(min(max(days + 60, 120), SINA_MAX_LEN))},
                       timeout=20, retries=2)
        arr = json.loads(txt) if isinstance(txt, str) else txt
    except Exception as exc:  # noqa: BLE001
        return _empty_kline(dict(attrs, error="新浪指数 K 线失败: %s" % exc))
    if not isinstance(arr, list) or not arr:
        return _empty_kline(dict(attrs, error="新浪指数 K 线返回空"))
    recs = [{"date": it.get("day"), "open": safe_float(it.get("open")),
             "high": safe_float(it.get("high")), "low": safe_float(it.get("low")),
             "close": safe_float(it.get("close")), "volume": safe_float(it.get("volume")),
             "amount": np.nan, "pct_chg": np.nan, "turnover": np.nan}
            for it in arr if isinstance(it, dict)]
    df = _sanitize(_df_of(recs, attrs))
    df.attrs.update(attrs)
    df.attrs["adjust"] = "none"
    return df


def _index_akshare(info: dict, days: int) -> pd.DataFrame:
    """akshare 指数日线（``stock_zh_index_daily``，实测 sh000001=8738 根）。"""
    attrs = {"source": "akshare", "requested_adjust": "none"}
    if _offline():
        return _empty_kline(dict(attrs, error="offline 模式跳过联网"))
    try:
        import akshare as ak
    except Exception as exc:  # noqa: BLE001
        return _empty_kline(dict(attrs, error="akshare 不可用: %s" % exc))
    sym = _sina_code(info)
    try:
        with _quiet():
            raw = retry_call(lambda: ak.stock_zh_index_daily(symbol=sym), attempts=2,
                             delay=0.8, default=None, label="ak_index")
    except Exception as exc:  # noqa: BLE001
        return _empty_kline(dict(attrs, error="akshare 指数日线失败: %s" % exc))
    if raw is None or len(raw) == 0:
        return _empty_kline(dict(attrs, error="akshare 指数日线返回空（%s）" % sym))
    cols = {str(c).lower(): c for c in raw.columns}
    recs = pd.DataFrame({
        "date": raw[cols.get("date", raw.columns[0])],
        "open": raw[cols["open"]] if "open" in cols else np.nan,
        "high": raw[cols["high"]] if "high" in cols else np.nan,
        "low": raw[cols["low"]] if "low" in cols else np.nan,
        "close": raw[cols["close"]] if "close" in cols else np.nan,
        "volume": raw[cols["volume"]] if "volume" in cols else np.nan,
    })
    recs["amount"] = np.nan
    recs["pct_chg"] = np.nan
    recs["turnover"] = np.nan
    df = _sanitize(_df_of(recs.to_dict("records"), attrs))
    df.attrs.update(attrs)
    df.attrs["adjust"] = "none"
    return df


def _index_tencent(info: dict, days: int) -> pd.DataFrame:
    """腾讯指数日线（指数忽略复权参数，键恒为 day/week/month）。"""
    attrs = {"source": "tencent", "requested_adjust": "none"}
    if _offline():
        return _empty_kline(dict(attrs, error="offline 模式跳过联网"))
    tx = _tx_code(info)
    collected: list[list] = []
    end = ""
    pages = 0
    want = int(days) + 5
    while pages < TENCENT_KLINE_MAX_PAGES and (len(collected) < want or pages == 0):
        rows = _tencent_fetch_raw(tx, "day", end, TENCENT_KLINE_MAX_COUNT, "")
        pages += 1
        if not rows:
            break
        fresh = [r for r in rows if not collected or str(r[0]) < end]
        collected = fresh + collected
        if not fresh or len(rows) < TENCENT_KLINE_MAX_COUNT:
            break
        end = str(rows[0][0])
    if not collected:
        return _empty_kline(dict(attrs, error="腾讯指数 K 线返回空（%s）" % tx))
    df = _tencent_rows_to_df(collected, tx)
    df = _tencent_volume_to_shares(df, info.get("prefix", ""))   # 指数同样是"手"
    df = _sanitize(df)
    df.attrs.update(attrs)
    df.attrs["adjust"] = "none"
    df.attrs["name"] = _TX_LAST_NAME.get(tx) or df.attrs.get("name")
    df.attrs["note"] = "指数成交量按腾讯口径为手，已×100 换成股"
    return df


def fetch_index(symbol="sh000001", days: int | None = None,
                force: bool = False) -> pd.DataFrame:
    """指数日线（新浪 → 东财 → akshare → 腾讯），返回结构与 :func:`fetch_kline` 一致。

    指数没有复权概念（``attrs["adjust"]="none"``）。``volume`` 单位为股
    （新浪/akshare 原生即股；腾讯为手，已×100）。``amount`` 由
    ``volume*(H+L+C)/3`` 估算并标注——**指数字段的量纲不可与个股直接比较**，
    建议只用于量能相对变化。
    """
    info = _norm_index_symbol(symbol)
    n_days = int(days) if days else _default_days("daily")
    key = cache_key("index", info.get("display"), n_days)

    hit = _cache_read(key, TTL_KLINE, force)
    if isinstance(hit, pd.DataFrame) and len(hit):
        out = _clone(hit)
        out.attrs["cached"] = True
        return out
    if _offline():
        return _empty_kline({"symbol": info.get("display"), "period": "daily",
                             "source": None, "adjust": "none", "rows": 0,
                             "fetched_at": _now_str(), "offline": True,
                             "error": "offline 模式（config.RUNTIME['offline']=True）"})

    order = [s for s in config.DATASOURCE_PRIORITY.get("index", ("sina", "eastmoney",
                                                                 "akshare", "tencent"))]
    impl = {"sina": _index_sina, "akshare": _index_akshare, "tencent": _index_tencent,
            "eastmoney": lambda i, d: _daily_eastmoney(i, "daily", d, "none")}
    need = max(1, int(n_days * ROWS_SUFFICIENT_RATIO))
    best: pd.DataFrame | None = None
    best_src = ""
    errors: list[str] = []
    tried: list[str] = []
    for i, name in enumerate(order):
        fn = impl.get(name)
        if fn is None:
            continue
        tried.append(name)
        try:
            df = fn(info, n_days)
        except Exception as exc:  # noqa: BLE001
            df = _empty_kline()
            errors.append("%s: %s: %s" % (name, type(exc).__name__, exc))
        if df is None or len(df) == 0:
            errors.append("%s: %s" % (name, (df.attrs.get("error") if df is not None else "空")))
            nxt = order[i + 1] if i + 1 < len(order) else "(无更多源)"
            log.info("index 源 %s 失败，降级到 %s", name, nxt)
            continue
        if best is None or len(df) > len(best):
            best, best_src = df, name
        if len(df) >= need:
            break
        log.info("index 源 %s 仅 %d 根(<%d)，降级到 %s", name, len(df), need,
                 order[i + 1] if i + 1 < len(order) else "(无更多源)")
    if best is None:
        warn = {"symbol": info.get("display"), "period": "daily", "source": None,
                "adjust": "none", "rows": 0, "fetched_at": _now_str(),
                "error": "所有指数量源均失败: " + "; ".join(errors[:6])}
        return _empty_kline(warn)
    be = best.attrs.get("error")
    out = _finalize_kline(best, info, "daily", "none", n_days, best_src,
                          requested_adjust="none", cached=False)
    if be:
        out.attrs["error"] = be
    if errors:
        out.attrs["degraded_from"] = errors
    _cache_write_if_ok(key, out, ok=len(out) > 0)
    return out


# --------------------------------------------------------------------------- #
# 实时快照
# --------------------------------------------------------------------------- #
def _parse_qt_time(raw: str) -> str | None:
    """腾讯时间字段归一化：``20261009161459`` / ``2026/10/09 16:08:14`` / ISO。"""
    s = str(raw or "").strip()
    if not s:
        return None
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", s)
    if m:
        return "%s-%s-%s %s:%s:%s" % m.groups()
    m = re.fullmatch(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})[ T](\d{1,2}):(\d{2}):(\d{2})", s)
    if m:
        return "%04d-%02d-%02d %02d:%02d:%02d" % tuple(int(x) for x in m.groups())
    return s


def _realtime_tencent(info: dict) -> dict:
    """腾讯实时快照（A 股完整映射；港美股只取实测同义的公共字段）。"""
    if _offline():
        return {}
    code = _tx_code(info)
    txt = http_get(TENCENT_QUOTE_URL + code, timeout=10, retries=2, encoding="gbk")
    if not isinstance(txt, str) or '="' not in txt:
        return {}
    body = txt.split('="', 1)[1].rsplit('"', 1)[0]
    if not body or body == "1":      # 未知代码返回 v_pv_none_match="1";
        return {}
    parts = body.split("~")
    if len(parts) < 40:
        return {}

    is_cn = bool(info.get("is_cn"))
    fmap = TENCENT_FIELD_MAP if is_cn else TENCENT_FOREIGN_FIELD_MAP
    raw: dict[str, Any] = {}
    for idx, (fname, _desc) in fmap.items():
        if idx < len(parts):
            raw[fname] = parts[idx]

    g = lambda k: safe_float(raw.get(k))                      # noqa: E731
    def _pos(k):                                              # noqa: E306
        v = g(k)
        return v if (v is not None and not math.isnan(v) and v > 0) else None

    price = _pos("price")
    prev_close = _pos("prev_close")
    high, low = _pos("high"), _pos("low")

    vol_lots = g("volume_lots") if is_cn else g("volume")
    volume = vol_lots * 100.0 if (is_cn and not math.isnan(vol_lots)) else \
        (None if math.isnan(vol_lots) else vol_lots)

    amount = None
    if is_cn:
        aw = g("amount_wan_precise")
        if math.isnan(aw):
            aw = g("amount_wan")
        if not math.isnan(aw):
            amount = aw * 1e4                                  # 万元 → 元
    else:
        if TENCENT_FOREIGN_AMOUNT_IDX < len(parts):
            a = safe_float(parts[TENCENT_FOREIGN_AMOUNT_IDX])
            amount = None if math.isnan(a) else a

    # 五档盘口：腾讯"价在前、量在后"，量单位=手 → 统一换成股
    bids, asks = [], []
    for lvl in range(1, 6):
        bp, bv = g("bid%d" % lvl), g("bid%d_lots" % lvl)
        ap, av = g("ask%d" % lvl), g("ask%d_lots" % lvl)
        if not math.isnan(bp) and bp > 0:
            bids.append([bp, None if math.isnan(bv) else bv * 100.0])
        if not math.isnan(ap) and ap > 0:
            asks.append([ap, None if math.isnan(av) else av * 100.0])

    limit_up, limit_down = _pos("limit_up"), _pos("limit_down")
    mc_yi = _pos("market_cap_yi")
    fmc_yi = _pos("float_market_cap_yi")
    total_shares, float_shares = _pos("total_shares"), _pos("float_shares")
    if not is_cn:                                              # 港美股股本单位为"股"
        idx = TENCENT_FOREIGN_SHARES_IDX.get(info.get("prefix"), (None, None))
        for slot, ii in (("float_shares", idx[0]), ("total_shares", idx[1])):
            if ii is not None and ii < len(parts):
                v = safe_float(parts[ii])
                if not math.isnan(v) and v > 0:
                    if slot == "float_shares":
                        float_shares = v
                    else:
                        total_shares = v

    # 未归类字段 → extra（便于后续挖掘，不做臆测命名）
    extra: dict[str, Any] = {}
    for i, v in enumerate(parts):
        if i in fmap or v == "":
            continue
        extra["f%d" % i] = v
    extra["bid_ask_unit"] = "股"
    extra["field_count"] = len(parts)
    extra["market_flag"] = parts[0] if parts else None
    if not is_cn:
        extra["layout"] = "foreign(hk/us) 字段布局与 A 股不同，仅映射实测同义索引"

    pct = g("pct_chg")
    return {
        "symbol": info.get("display"), "name": parts[1] or None,
        "price": price, "prev_close": prev_close, "open": _pos("open"),
        "high": high, "low": low, "volume": volume, "amount": amount,
        "pct_chg": None if math.isnan(pct) else pct,
        "amplitude": _pos("amplitude"), "turnover": _pos("turnover"),
        "volume_ratio": _pos("volume_ratio"), "pe_ttm": _pos("pe_ttm"),
        "pe_static": _pos("pe_static"), "pb": _pos("pb"),
        "market_cap": None if mc_yi is None else mc_yi * 1e8,
        "float_market_cap": None if fmc_yi is None else fmc_yi * 1e8,
        "total_shares": total_shares, "float_shares": float_shares,
        "limit_up": limit_up, "limit_down": limit_down,
        "bid_ask": {"bids": bids, "asks": asks},
        "time": _parse_qt_time(raw.get("time")), "source": "tencent",
        "extra": extra,
    }


def _realtime_sina(info: dict) -> dict:
    """新浪实时快照（A 股；五档为"量在前、价在后"，量单位=股）。"""
    if _offline():
        return {}
    code = _sina_code(info)
    txt = http_get(SINA_QUOTE_URL + code, timeout=10, retries=2, encoding="gbk",
                   headers=_SINA_HEADERS)
    if not isinstance(txt, str) or '="' not in txt:
        return {}
    body = txt.split('="', 1)[1].rsplit('"', 1)[0]
    if not body.strip():
        return {}
    parts = body.split(",")
    if len(parts) < 32:
        return {}
    g = lambda i: safe_float(parts[i]) if i < len(parts) else float("nan")   # noqa: E731

    def _pos(v):
        return v if (v is not None and not math.isnan(v) and v > 0) else None

    price, prev_close = _pos(g(3)), _pos(g(2))
    bids, asks = [], []
    for lvl in range(5):
        bv, bp = g(10 + lvl * 2), g(11 + lvl * 2)
        av, ap = g(20 + lvl * 2), g(21 + lvl * 2)
        if not math.isnan(bp) and bp > 0:
            bids.append([bp, None if math.isnan(bv) else bv])
        if not math.isnan(ap) and ap > 0:
            asks.append([ap, None if math.isnan(av) else av])

    volume = g(8)                       # 股
    amount = g(9)                       # 元
    pct = float("nan")
    if price and prev_close:
        pct = (price - prev_close) / prev_close * 100.0
    high, low = _pos(g(4)), _pos(g(5))
    amp = (high - low) / prev_close * 100.0 if (high and low and prev_close) else float("nan")
    date_s = parts[30] if len(parts) > 30 else ""
    time_s = parts[31] if len(parts) > 31 else ""
    return {
        "symbol": info.get("display"), "name": parts[0] or None,
        "price": price, "prev_close": prev_close, "open": _pos(g(1)),
        "high": high, "low": low,
        "volume": None if math.isnan(volume) else volume,
        "amount": None if math.isnan(amount) else amount,
        "pct_chg": None if math.isnan(pct) else round(pct, 4),
        "amplitude": None if math.isnan(amp) else round(amp, 4),
        "turnover": None, "volume_ratio": None, "pe_ttm": None, "pe_static": None,
        "pb": None, "market_cap": None, "float_market_cap": None,
        "total_shares": None, "float_shares": None,
        "limit_up": None, "limit_down": None,
        "bid_ask": {"bids": bids, "asks": asks},
        "time": ("%s %s" % (date_s, time_s)).strip() or None,
        "source": "sina",
        "extra": {"status": parts[32] if len(parts) > 32 else None,
                  "after_hours": parts[33] if len(parts) > 33 else None,
                  "field_count": len(parts), "bid_ask_unit": "股",
                  "note": "新浪实时不含 PE/PB/市值/涨跌停/换手率"},
    }


def _yf_fast_info(info: dict) -> dict:
    """yfinance 兜底快照（港美股用）。"""
    yf = _yf_module()
    if yf is None:
        return {}
    sym = info.get("yf_symbol") or info.get("code")
    try:
        t = yf.Ticker(sym)
        fi = dict(getattr(t, "fast_info", {}) or {})
    except Exception as exc:  # noqa: BLE001
        log.info("realtime yfinance 失败：%s", exc)
        return {}
    if not fi:
        return {}

    def _pick(*keys):
        for k in keys:
            v = fi.get(k)
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isnan(f) and f > 0:
                return f
        return None

    price = _pick("lastPrice", "last_price", "regularMarketPrice")
    prev = _pick("previousClose", "previous_close", "regularMarketPreviousClose")
    pct = None
    if price and prev:
        pct = round((price - prev) / prev * 100.0, 4)
    return {
        "symbol": info.get("display"), "name": None, "price": price, "prev_close": prev,
        "open": _pick("open"), "high": _pick("dayHigh", "day_high"),
        "low": _pick("dayLow", "day_low"),
        "volume": _pick("lastVolume", "last_volume"), "amount": None,
        "pct_chg": pct, "amplitude": None, "turnover": None, "volume_ratio": None,
        "pe_ttm": None, "pe_static": None, "pb": None, "market_cap": None,
        "float_market_cap": None, "total_shares": _pick("shares"),
        "float_shares": _pick("shares"),
        "limit_up": None, "limit_down": None, "bid_ask": {"bids": [], "asks": []},
        "time": None, "source": "yfinance",
        "extra": {"fast_info": {k: str(v) for k, v in fi.items()}},
    }


def _finalize_realtime(info: dict, snap: dict, source: str) -> dict:
    """补齐派生指标与字段完整性（缺失一律 None，不用 0 冒充）。"""
    price = snap.get("price")
    high, low = snap.get("high"), snap.get("low")
    pos = None
    if price and high and low and high > low:
        pos = round(min(max((price - low) / (high - low), 0.0), 1.0), 4)
    vs_limit = None
    if price and snap.get("limit_up"):
        vs_limit = round((snap["limit_up"] - price) / price * 100.0, 4)

    out = {
        "symbol": snap.get("symbol") or info.get("display"),
        "name": snap.get("name"), "price": price,
        "prev_close": snap.get("prev_close"), "open": snap.get("open"),
        "high": high, "low": low, "volume": snap.get("volume"),
        "amount": snap.get("amount"), "pct_chg": snap.get("pct_chg"),
        "amplitude": snap.get("amplitude"), "turnover": snap.get("turnover"),
        "volume_ratio": snap.get("volume_ratio"), "pe_ttm": snap.get("pe_ttm"),
        "pe_static": snap.get("pe_static"), "pb": snap.get("pb"),
        "market_cap": snap.get("market_cap"),
        "float_market_cap": snap.get("float_market_cap"),
        "total_shares": snap.get("total_shares"),
        "float_shares": snap.get("float_shares"),
        "limit_up": snap.get("limit_up"), "limit_down": snap.get("limit_down"),
        "bid_ask": snap.get("bid_ask") or {"bids": [], "asks": []},
        "time": snap.get("time"), "source": source,
        "extra": snap.get("extra") or {},
        # ---- 派生指标 ----
        "position_in_day": pos,          # 现价在当日高低区间的位置 0~1
        "vs_limit_up_pct": vs_limit,     # 距涨停还有多少 %
        "code": info.get("code"), "prefix": info.get("prefix"),
        "currency": "CNY" if info.get("is_cn") else None,
        "fetched_at": _now_str(),
    }
    return out


def fetch_realtime(symbol, force: bool = False) -> dict:
    """实时快照（腾讯 → 新浪；港美股走 yfinance → 腾讯公共字段）。

    Returns
    -------
    dict
        键固定包含：``symbol,name,price,prev_close,open,high,low,volume,amount,
        pct_chg,amplitude,turnover,volume_ratio,pe_ttm,pe_static,pb,market_cap,
        float_market_cap,total_shares,float_shares,limit_up,limit_down,bid_ask,
        time,source,extra`` 以及派生量 ``position_in_day``（现价在当日高低区间的
        位置 0~1）、``vs_limit_up_pct``（距涨停还有多少 %）。
        **拿不到的字段一律 None**（绝不用 0 冒充）；``volume`` 单位股、``amount``
        单位元、``market_cap`` 单位元、``bid_ask`` 量为股。全部失败返回 ``{}``。
        ``extra`` 内含腾讯返回中未归类的原始字段（``fNN``）。
    """
    info = normalize_symbol(symbol)
    key = cache_key("realtime", info.get("display"))
    if not force:
        hit = cache_get(key, TTL_REALTIME)
        if isinstance(hit, dict) and hit:
            out = dict(hit)
            out["cached"] = True
            return out
    if _offline():
        return {}

    order = list(config.DATASOURCE_PRIORITY.get("realtime", ("tencent", "sina")))
    if info.get("is_cn"):
        impl = {"tencent": _realtime_tencent, "sina": _realtime_sina}
    else:                                   # 港美股：yfinance 优先，腾讯公共字段兜底
        impl = {"yfinance": _yf_fast_info, "tencent": _realtime_tencent}
        order = ["yfinance", "tencent"]
    snap: dict = {}
    source = ""
    for i, name in enumerate(order):
        fn = impl.get(name)
        if fn is None:
            continue
        try:
            got = fn(info)
        except Exception as exc:  # noqa: BLE001
            log.info("realtime 源 %s 失败：%s", name, exc)
            got = {}
        if got and got.get("price"):
            snap, source = got, name
            break
        log.info("realtime 源 %s 失败，降级到 %s", name,
                 order[i + 1] if i + 1 < len(order) else "(无更多源)")
    if not snap:
        return {}
    out = _finalize_realtime(info, snap, source)
    if out.get("price"):
        _cache_write_if_ok(key, out, ok=True)
    return out


def fetch_index_realtime(symbol="sh000001", force: bool = False) -> dict:
    """指数实时快照（腾讯 → 新浪）。字段结构与 :func:`fetch_realtime` 一致，
    指数没有涨跌停/PE/PB（对应键为 None）。"""
    info = _norm_index_symbol(symbol)
    key = cache_key("index_rt", info.get("display"))
    if not force:
        hit = cache_get(key, TTL_REALTIME)
        if isinstance(hit, dict) and hit:
            out = dict(hit)
            out["cached"] = True
            return out
    if _offline():
        return {}
    snap, source = {}, ""
    for name, fn in (("tencent", _realtime_tencent), ("sina", _realtime_sina)):
        try:
            got = fn(info)
        except Exception as exc:  # noqa: BLE001
            log.info("index realtime 源 %s 失败：%s", name, exc)
            got = {}
        if got and got.get("price"):
            snap, source = got, name
            break
        log.info("index realtime 源 %s 失败，降级到 %s", name,
                 "sina" if name == "tencent" else "(无更多源)")
    if not snap:
        return {}
    out = _finalize_realtime(info, snap, source)
    _cache_write_if_ok(key, out, ok=True)
    return out


# --------------------------------------------------------------------------- #
# 代码表与代码解析
# --------------------------------------------------------------------------- #
_STOCK_LIST_COLUMNS = ["code", "name", "display", "prefix", "market", "secid",
                       "yf_symbol", "board"]
_SEARCH_LOCK = threading.Lock()
_SEARCH_CACHE: dict[str, Any] = {"key": None, "frame": None, "index": None}


def _stock_list_eastmoney(limit: int = 6000) -> pd.DataFrame:
    """东财 clist 全市场代码表（本机网络不可达时返回空）。"""
    if _offline():
        return pd.DataFrame()
    def _call():
        txt = http_get(EM_CLIST_URL, params={
            "pn": 1, "pz": int(limit), "po": 1, "np": 1, "fltt": 2, "invt": 2,
            "fid": "f12",
            "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
            "fields": "f12,f14,f13",
        }, timeout=20, retries=1)
        return json.loads(txt) if isinstance(txt, str) else txt
    data = retry_call(_call, attempts=3, delay=0.6, default=None, label="em_clist")
    if not isinstance(data, dict):
        return pd.DataFrame()
    diff = (data.get("data") or {}).get("diff") or []
    if isinstance(diff, dict):
        diff = list(diff.values())
    rows = [{"code": str(it.get("f12") or "").zfill(6), "name": str(it.get("f14") or "")}
            for it in diff if isinstance(it, dict) and it.get("f12")]
    return pd.DataFrame(rows)


def _stock_list_akshare() -> pd.DataFrame:
    """akshare 全市场代码表（实测 5573 条，含沪深主板/科创/创业/北交所 92xxxx）。"""
    if _offline():
        return pd.DataFrame()
    try:
        import akshare as ak
    except Exception as exc:  # noqa: BLE001
        log.info("stock_list akshare 不可用：%s", exc)
        return pd.DataFrame()
    try:
        with _quiet():
            raw = retry_call(lambda: ak.stock_info_a_code_name(), attempts=2, delay=0.8,
                             default=None, label="ak_code_name")
    except Exception as exc:  # noqa: BLE001
        log.info("stock_list akshare 失败：%s", exc)
        return pd.DataFrame()
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    cols = {str(c).lower(): c for c in raw.columns}
    if "code" not in cols or "name" not in cols:
        return pd.DataFrame()
    return pd.DataFrame({"code": raw[cols["code"]].astype(str).str.strip().str.zfill(6),
                         "name": raw[cols["name"]].astype(str).str.strip()})


def _build_stock_frame(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    """原始 code/name → 带 display/prefix/secid/yf_symbol/board 的完整表。"""
    if raw is None or len(raw) == 0:
        return _empty_df(_STOCK_LIST_COLUMNS, {"source": source, "rows": 0})
    raw = raw.dropna(subset=["code"])
    raw = raw[raw["code"].astype(str).str.fullmatch(r"\d{6}")]
    raw = raw.drop_duplicates(subset=["code"], keep="first")
    rows = []
    for code, name in zip(raw["code"].astype(str), raw["name"].astype(str)):
        info = normalize_symbol(code)
        rows.append({"code": code, "name": name, "display": info["display"],
                     "prefix": info["prefix"], "market": info["market"],
                     "secid": info["secid"], "yf_symbol": info["yf_symbol"],
                     "board": _board_of(code)})
    df = pd.DataFrame(rows, columns=_STOCK_LIST_COLUMNS)
    df.attrs.update({"source": source, "rows": int(len(df)), "fetched_at": _now_str()})
    return df


def _board_of(code: str) -> str:
    from .utils import board_of
    return board_of(code)


_STOCK_LIST_LOCK = threading.Lock()


def fetch_stock_list(force: bool = False) -> pd.DataFrame:
    """A 股全市场代码表（akshare 优先，东财 clist 备选）。

    Returns
    -------
    pandas.DataFrame
        列 ``["code","name","display","prefix","market","secid","yf_symbol","board"]``；
        ``code`` 为 6 位字符串。失败返回同列表格（``attrs["error"]`` 说明原因）。
        用于代码→名称、名称→代码互查（:func:`resolve_symbol` / :func:`search_symbol`）。
        整表需要拉 18 页（约 6~19s），因此用**单飞锁**避免并发重复拉取，
        并缓存 ``cache_ttl_daily``（24h）。
    """
    key = cache_key("stocklist", "all")
    hit = _cache_read(key, TTL_LIST, force)
    if isinstance(hit, pd.DataFrame) and len(hit):
        return _clone(hit)
    if _offline():
        return _empty_df(_STOCK_LIST_COLUMNS, {"source": None, "rows": 0,
                                               "error": "offline 模式跳过联网"})
    with _STOCK_LIST_LOCK:                 # 单飞：并发调用只真正拉一次
        hit = _cache_read(key, TTL_LIST, force)      # 等锁期间别人可能已经拉好
        if isinstance(hit, pd.DataFrame) and len(hit):
            return _clone(hit)
        raw, source = _stock_list_akshare(), "akshare"
        if len(raw) == 0:
            log.info("stock_list 源 akshare 失败，降级到 eastmoney")
            raw, source = _stock_list_eastmoney(), "eastmoney"
        if len(raw) == 0:
            return _empty_df(_STOCK_LIST_COLUMNS,
                             {"source": None, "rows": 0, "fetched_at": _now_str(),
                              "error": "所有代码表源均失败"})
        df = _build_stock_frame(raw, source)
        if len(df):
            _cache_write_if_ok(key, df, ok=True)
        return df


def _search_key(text: str) -> str:
    """检索归一化键：NFKC（全角→半角）+ 去空白 + 大写。"""
    s = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"\s+", "", s).upper()


def _cached_stock_list() -> pd.DataFrame | None:
    """只读缓存的代码表（绝不联网）；未缓存返回 ``None``。"""
    hit = cache_get(cache_key("stocklist", "all"), TTL_LIST)
    return hit if isinstance(hit, pd.DataFrame) and len(hit) else None


def _search_table(allow_fetch: bool = True) -> tuple[pd.DataFrame | None, dict | None]:
    """缓存代码表 + 归一化名称/代码索引（``allow_fetch=False`` 时不发起联网拉取）。"""
    df = fetch_stock_list() if allow_fetch else _cached_stock_list()
    if df is None or len(df) == 0:
        return None, None
    stamp = (len(df), df.attrs.get("fetched_at"), id(df))
    with _SEARCH_LOCK:
        if _SEARCH_CACHE["key"] == stamp and _SEARCH_CACHE["index"] is not None:
            return _SEARCH_CACHE["frame"], _SEARCH_CACHE["index"]
        work = df.copy()
        work["_nk"] = work["name"].map(_search_key)
        work["_ck"] = work["code"].astype(str)
        index = {}
        for ck, nk, disp, name, code, board in zip(work["_ck"], work["_nk"],
                                                   work["display"], work["name"],
                                                   work["code"], work["board"]):
            index.setdefault(nk, (code, name, disp, board))
        _SEARCH_CACHE.update({"key": stamp, "frame": work, "index": index})
        return work, index


def _row_to_symbol(code: str, name: str | None, matched_by: str) -> dict:
    info = normalize_symbol(code)
    return {"code": info["code"], "name": name, "display": info["display"],
            "prefix": info["prefix"], "secid": info["secid"],
            "yf_symbol": info["yf_symbol"], "market": info["market"],
            "is_cn": info["is_cn"], "matched_by": matched_by}


def search_symbol(keyword, limit: int = 10, allow_fetch: bool = True) -> list:
    """在 A 股代码表里做模糊检索（代码精确 → 名称精确 → 名称前缀 → 名称包含）。

    Returns
    -------
    list[dict]
        元素与 :func:`resolve_symbol` 同结构（可直接喂给股票池），
        按相关度排序，最多 ``limit`` 条；代码表不可用时返回 ``[]``。
        ``allow_fetch=False`` 时只用已缓存的代码表，不联网。
    """
    kw = _search_key(keyword)
    if not kw:
        return []
    frame, _index = _search_table(allow_fetch=allow_fetch)
    if frame is None or len(frame) == 0:
        return []
    try:
        lim = max(1, int(limit))
    except (TypeError, ValueError):
        lim = 10

    exact_code = frame[frame["_ck"] == kw]
    exact_name = frame[frame["_nk"] == kw]
    starts = frame[frame["_nk"].str.startswith(kw, na=False)]
    contains = frame[frame["_nk"].str.contains(re.escape(kw), na=False)]
    code_pref = frame[frame["_ck"].str.startswith(kw, na=False)]

    out: list[dict] = []
    seen: set[str] = set()
    for part, tag in ((exact_code, "code"), (exact_name, "name_exact"),
                      (starts, "name_prefix"), (contains, "name_fuzzy"),
                      (code_pref, "code_prefix")):
        for _i, r in part.iterrows():
            if r["code"] in seen:
                continue
            seen.add(r["code"])
            out.append(_row_to_symbol(r["code"], r["name"], tag))
            if len(out) >= lim:
                return out
    return out


def resolve_symbol(query, lookup_name: bool = True) -> dict:
    """把任意写法的标的解析成统一结构（代码 / 中文名 / 模糊名 / 港美股）。

    支持 ``"600519"`` / ``"sh600519"`` / ``"600519.SH"`` / ``"贵州茅台"`` / ``"茅台"``
    / ``"00700.HK"`` / ``"AAPL"``。

    Parameters
    ----------
    query : str
        任意写法的代码或名称。
    lookup_name : bool
        是否允许为查名称而拉取全市场代码表（首次约 6~19s，之后走 24h 缓存）。
        设为 ``False`` 时只用已缓存的代码表补齐 ``name``（体检等对耗时敏感的场景用）。

    Returns
    -------
    dict
        ``{"code","name","display","prefix","secid","yf_symbol","market",
        "is_cn","matched_by"}``；名称查不到时 ``name`` 为 ``None``，其余字段
        来自 :func:`core.utils.normalize_symbol`，``matched_by`` 说明命中方式
        （``code`` / ``code_prefix`` / ``code_suffix`` / ``name_exact`` /
        ``name_prefix`` / ``name_fuzzy`` / ``us_ticker`` / ``fallback``）。
    """
    q = str(query or "").strip()
    if not q:
        base = normalize_symbol("")
        return {"code": "", "name": None, "display": "", "prefix": "", "secid": "",
                "yf_symbol": "", "market": "", "is_cn": False, "matched_by": "empty"}

    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", q))
    # ---- 中文（含中文简称）→ 走代码表检索 ----
    if has_cjk:
        hits = search_symbol(q, limit=1, allow_fetch=lookup_name)
        if hits:
            return hits[0]
        base = normalize_symbol(re.sub(r"[^\x00-\x7f]", "", q) or q)
        return {"code": base["code"], "name": None, "display": base["display"],
                "prefix": base["prefix"], "secid": base["secid"],
                "yf_symbol": base["yf_symbol"], "market": base["market"],
                "is_cn": base["is_cn"], "matched_by": "fallback"}

    # ---- 代码 / 字母代码 ----
    if re.fullmatch(r"(SH|SZ|BJ|HK|US)\.?\w{1,10}", q.upper()):
        matched = "code_prefix"
    elif re.fullmatch(r"\w{1,10}\.(SH|SS|SZ|XSHE|XSHG|BJ|HK|US)", q.upper()):
        matched = "code_suffix"
    elif re.fullmatch(r"\d{4,6}", q):
        matched = "code"
    elif re.fullmatch(r"[A-Za-z][A-Za-z.\-]{0,9}", q):
        matched = "us_ticker"
    else:
        matched = "fallback"

    base = normalize_symbol(q)
    name = None
    if matched != "us_ticker":
        _frame, index = _search_table(allow_fetch=lookup_name)
        if index:
            hit = index.get(base["code"].zfill(6)) or index.get(base["code"])
            if hit:
                name = hit[1]
    if name is None and matched == "fallback":
        hits = search_symbol(q, limit=1, allow_fetch=lookup_name)
        if hits:
            return hits[0]
    return {"code": base["code"], "name": name, "display": base["display"],
            "prefix": base["prefix"], "secid": base["secid"],
            "yf_symbol": base["yf_symbol"], "market": base["market"],
            "is_cn": base["is_cn"], "matched_by": matched}


# --------------------------------------------------------------------------- #
# 板块
# --------------------------------------------------------------------------- #
_BOARD_TYPE_CACHE: dict[str, Any] = {"map": None, "ts": 0.0}
_BOARD_TYPE_LOCK = threading.Lock()


def _board_type_map(force: bool = False) -> dict[str, str]:
    """东财板块代码 → 类型（行业/概念/地域）。本机 clist 不可达时返回 {}。"""
    with _BOARD_TYPE_LOCK:
        if not force and _BOARD_TYPE_CACHE["map"] is not None \
                and time.time() - _BOARD_TYPE_CACHE["ts"] < TTL_LIST:
            return _BOARD_TYPE_CACHE["map"]
    mapping: dict[str, str] = {}
    if not _offline():
        for t, label in EM_BOARD_TYPE_FS.items():
            def _call(t=t):
                txt = http_get(EM_CLIST_URL, params={
                    "pn": 1, "pz": 500, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                    "fid": "f12", "fs": "m:90+t:%s" % t, "fields": "f12,f14",
                }, timeout=12, retries=1)
                return json.loads(txt) if isinstance(txt, str) else txt
            data = retry_call(_call, attempts=2, delay=0.5, default=None,
                              label="em_board_type_%s" % t)
            if not isinstance(data, dict):
                continue
            diff = (data.get("data") or {}).get("diff") or []
            if isinstance(diff, dict):
                diff = list(diff.values())
            for it in diff:
                code = str((it or {}).get("f12") or "")
                if code:
                    mapping[code] = label
    with _BOARD_TYPE_LOCK:
        _BOARD_TYPE_CACHE["map"] = mapping
        _BOARD_TYPE_CACHE["ts"] = time.time()
    return mapping


def _guess_board_type(code: str, name: str, mapping: dict[str, str]) -> str:
    """板块类型：优先精确映射；退化为名称启发式（东财地域板块名以"板块"结尾）。"""
    if code in mapping:
        return mapping[code]
    if str(name).endswith("板块"):
        return "地域"
    return BOARD_TYPE_UNKNOWN


def fetch_sector_list(symbol, force: bool = False) -> list:
    """个股所属板块（东财 slist，``spt=3``）。

    Returns
    -------
    list[dict]
        ``[{"code","name","type","pct","main_net"}, ...]``：
        ``type`` ∈ {行业, 概念, 地域}（无法确定时为 ``"板块"``）；
        ``pct`` 为板块涨跌幅(%)，``main_net`` 为主力净流入(元)。
        该域名间歇失败，已用 ``retry_call(attempts=3)``；失败返回 ``[]``。
    """
    info = normalize_symbol(symbol)
    key = cache_key("sector", info.get("display"))
    if not force:
        hit = cache_get(key, TTL_KLINE)
        if isinstance(hit, list) and hit:
            return list(hit)
    if _offline():
        return []

    def _call():
        txt = http_get(EM_SLIST_URL, params={
            "secid": info.get("secid"), "spt": 3, "np": 1, "fltt": 2, "invt": 2,
            "fields": "f12,f13,f14,f3,f62,f184", "po": 1, "pn": 1, "pz": 30,
            "fid": "f62",
        }, timeout=12, retries=1)
        return json.loads(txt) if isinstance(txt, str) else txt

    data = retry_call(_call, attempts=3, delay=0.6, default=None, label="em_slist")
    if not isinstance(data, dict):
        log.info("sector 源 eastmoney 失败（slist 不可达），返回空列表")
        return []
    diff = (data.get("data") or {}).get("diff") or []
    if isinstance(diff, dict):          # 未带 np=1 时 diff 是 dict（实测）
        diff = list(diff.values())
    mapping = _board_type_map()
    out = []
    for it in diff:
        if not isinstance(it, dict):
            continue
        code = str(it.get("f12") or "")
        name = str(it.get("f14") or "")
        if not code.startswith("BK"):
            continue
        pct = safe_float(it.get("f3"))
        net = safe_float(it.get("f62"))
        out.append({"code": code, "name": name,
                    "type": _guess_board_type(code, name, mapping),
                    "pct": None if math.isnan(pct) else pct,
                    "main_net": None if math.isnan(net) else net})
    if out:
        _cache_write_if_ok(key, out, ok=True)
    return out


_SECTOR_MEMBER_COLUMNS = ["代码", "名称", "现价", "涨跌幅", "PE", "总市值", "PB"]


def fetch_sector_members(board_code, force: bool = False,
                         board_name: str | None = None) -> pd.DataFrame:
    """板块成分股（东财 clist ``fs=b:BKxxxx``；akshare 板块成分作为兜底）。

    Parameters
    ----------
    board_code : str
        板块代码，如 ``BK0475``。
    board_name : str | None
        板块中文名（可选）。东财 clist 不可达时，用它走 akshare 的
        ``stock_board_industry_cons_em`` / ``stock_board_concept_cons_em`` 兜底。

    Returns
    -------
    pandas.DataFrame
        列 ``["代码","名称","现价","涨跌幅","PE","总市值","PB"]``；失败返回空表
        （``attrs["error"]`` 说明原因）。
    """
    code = str(board_code or "").strip().upper()
    key = cache_key("sector_members", code)
    if not force:
        hit = cache_get(key, TTL_KLINE)
        if isinstance(hit, pd.DataFrame) and len(hit):
            return _clone(hit)
    if _offline():
        return _empty_df(_SECTOR_MEMBER_COLUMNS, {"error": "offline 模式跳过联网",
                                                  "rows": 0, "board_code": code})

    def _call():
        txt = http_get(EM_CLIST_URL, params={
            "pn": 1, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2, "fid": "f3",
            "fs": "b:%s" % code, "fields": EM_CLIST_FIELDS,
        }, timeout=12, retries=1)
        return json.loads(txt) if isinstance(txt, str) else txt

    data = retry_call(_call, attempts=3, delay=0.6, default=None, label="em_clist_board")
    rows, source = [], "eastmoney"
    if isinstance(data, dict):
        diff = (data.get("data") or {}).get("diff") or []
        if isinstance(diff, dict):
            diff = list(diff.values())
        for it in diff:
            if not isinstance(it, dict):
                continue
            rows.append({"代码": str(it.get("f12") or "").zfill(6),
                         "名称": it.get("f14"),
                         "现价": safe_float(it.get("f2")),
                         "涨跌幅": safe_float(it.get("f3")),
                         "PE": safe_float(it.get("f9")),
                         "总市值": safe_float(it.get("f20")),
                         "PB": safe_float(it.get("f23"))})

    if not rows and board_name:                       # akshare 兜底（需要板块中文名）
        try:
            import akshare as ak
            with _quiet():
                raw = retry_call(lambda: ak.stock_board_industry_cons_em(symbol=board_name),
                                 attempts=1, default=None, label="ak_board_cons")
                if raw is None or len(raw) == 0:
                    raw = retry_call(lambda: ak.stock_board_concept_cons_em(symbol=board_name),
                                     attempts=1, default=None, label="ak_board_cons2")
            if raw is not None and len(raw):
                cols = {str(c): c for c in raw.columns}
                for _i, r in raw.iterrows():
                    rows.append({"代码": str(r.get(cols.get("代码", ""), "")).zfill(6),
                                 "名称": r.get(cols.get("名称", ""), None),
                                 "现价": safe_float(r.get(cols.get("最新价", ""), np.nan)),
                                 "涨跌幅": safe_float(r.get(cols.get("涨跌幅", ""), np.nan)),
                                 "PE": safe_float(r.get(cols.get("市盈率-动态", ""), np.nan)),
                                 "总市值": safe_float(r.get(cols.get("总市值", ""), np.nan)),
                                 "PB": safe_float(r.get(cols.get("市净率", ""), np.nan))})
                source = "akshare"
        except Exception as exc:  # noqa: BLE001
            log.info("sector_members akshare 兜底失败：%s", exc)

    if not rows:
        return _empty_df(_SECTOR_MEMBER_COLUMNS,
                         {"board_code": code, "rows": 0, "source": None,
                          "fetched_at": _now_str(),
                          "error": "板块成分获取失败（东财 clist 不可达%s）"
                                   % ("，且未提供 board_name" if not board_name else
                                      "，akshare 兜底也失败")})
    df = pd.DataFrame(rows, columns=_SECTOR_MEMBER_COLUMNS)
    df.attrs.update({"board_code": code, "source": source, "rows": int(len(df)),
                     "fetched_at": _now_str()})
    _cache_write_if_ok(key, df, ok=True)
    return df


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #
def clear_cache() -> int:
    """清空两级缓存（内存 + 磁盘），返回删除的缓存文件数。"""
    with _SEARCH_LOCK:
        _SEARCH_CACHE.update({"key": None, "frame": None, "index": None})
    with _BOARD_TYPE_LOCK:
        _BOARD_TYPE_CACHE.update({"map": None, "ts": 0.0})
    n = cache_clear()
    log.info("datasource 缓存已清空，删除文件 %d 个", n)
    return int(n)


# --------------------------------------------------------------------------- #
# 健康度体检
# --------------------------------------------------------------------------- #
def _run_parallel(tasks: dict[str, Callable[[], Any]], timeout: float) -> dict:
    """用守护线程并发执行任务，超过 ``timeout`` 未完成的项直接丢弃（记为超时）。"""
    results: dict[str, Any] = {}
    lock = threading.Lock()

    def _worker(name: str, fn: Callable[[], Any]) -> None:
        t0 = time.time()
        try:
            val = fn()
            with lock:
                results[name] = (val, int((time.time() - t0) * 1000), None)
        except Exception as exc:  # noqa: BLE001
            with lock:
                results[name] = (None, int((time.time() - t0) * 1000),
                                 "%s: %s" % (type(exc).__name__, exc))

    threads = [threading.Thread(target=_worker, args=(n, f), daemon=True,
                                name="health-" + n) for n, f in tasks.items()]
    for t in threads:
        t.start()
    deadline = time.time() + max(1.0, float(timeout))
    for t in threads:
        t.join(max(0.05, deadline - time.time()))
    return results


def _with_deadline(fn: Callable[[], Any], seconds: float, label: str = "") -> Any:
    """在守护线程里跑 ``fn``，超过 ``seconds`` 就放弃等待并抛 TimeoutError。

    被放弃的线程会继续在后台跑完（因此不会浪费：例如代码表拉完会自己写进缓存），
    但**不会**再拖住调用方——用于给"整表 18 页"这类慢任务设独立预算。
    """
    box: dict = {}

    def _run() -> None:
        try:
            box["v"] = fn()
        except Exception as exc:  # noqa: BLE001
            box["e"] = exc

    th = threading.Thread(target=_run, daemon=True, name="deadline-" + (label or "task"))
    th.start()
    th.join(max(0.05, float(seconds)))
    if "e" in box:
        raise box["e"]
    if "v" not in box:
        raise TimeoutError("%s 超过 %.0fs 仍未返回（后台继续执行，下次调用应命中缓存）"
                           % (label or "任务", seconds))
    return box["v"]


def _health_item(res: Any, default_error: str = "timeout") -> dict:
    """把 ``_run_parallel`` 的原始结果转成统一报告项。"""
    if res is None:
        return {"ok": False, "rows": None, "last_date": None, "ms": None,
                "error": default_error}
    val, ms, err = res
    if err:
        return {"ok": False, "rows": None, "last_date": None, "ms": ms, "error": err}
    rows, last = (val if isinstance(val, (tuple, list)) and len(val) == 2 else (val, None))
    return {"ok": bool(rows), "rows": int(rows or 0), "last_date": last, "ms": ms,
            "error": None if rows else "空结果"}


def data_health_check(symbol="600519", timeout: float = 20.0) -> dict:
    """并发体检所有数据源，返回逐项报告与总体结论。

    ``timeout`` 为总耗时上限（秒，默认 20）；超时的项会记为
    ``ok=False, error="timeout"``，因此函数**总是**按时返回。

    Returns
    -------
    dict
        ``{"symbol","checked_at","items":{名称:{ok,rows,last_date,ms,error}},
        "overall":"healthy"/"degraded"/"offline","notes":[...]}``
        条目包括：``daily_sina`` / ``daily_tencent`` / ``daily_eastmoney`` /
        ``daily_akshare`` / ``daily_yfinance`` / ``realtime_tencent`` /
        ``realtime_sina`` / ``index_sina`` / ``index_akshare`` / ``index_tencent`` /
        ``stock_list_akshare`` / ``stock_list_eastmoney`` / ``sector_eastmoney`` /
        ``resolve_symbol``。
        除代码表（走缓存以免每次都拉 18 页）外，其余项都用 ``force=True``
        **真实联网**测量，不走缓存。
    """
    info = normalize_symbol(symbol)
    disp = info.get("display") or "sh600519"

    def _kline_task(src: str):
        def _t():
            df = _KLINE_SOURCES[src][0](info, "daily", 60, "qfq")
            if df is None or len(df) == 0:
                raise RuntimeError((df.attrs.get("error") if df is not None else "") or "空结果")
            return len(df), str(pd.to_datetime(df["date"].iloc[-1]).date())
        return _t

    def _rt_task(src: str, sym: dict = None):
        def _t():
            snap = (_realtime_tencent if src == "tencent" else _realtime_sina)(sym or info)
            if not snap or not snap.get("price"):
                raise RuntimeError("未取到现价")
            return 1, str(snap.get("time") or "")
        return _t

    def _index_task(src: str):
        idx = _norm_index_symbol(config.RUNTIME.get("index_symbol", "sh000001"))

        def _t():
            fn = {"sina": _index_sina, "akshare": _index_akshare,
                  "tencent": _index_tencent}[src]
            df = fn(idx, 60)
            if df is None or len(df) == 0:
                raise RuntimeError((df.attrs.get("error") if df is not None else "") or "空结果")
            return len(df), str(pd.to_datetime(df["date"].iloc[-1]).date())
        return _t

    def _list_task(src: str):
        def _t():
            if src != "akshare":
                df = _stock_list_eastmoney()
                if df is None or len(df) == 0:
                    raise RuntimeError("代码表为空")
                return len(df), None
            # akshare 代码表是整表 18 页（冷启动 6~19s），不能拖累体检：
            # 已缓存 → 秒回；未缓存 → 起一个后台线程预热，本项如实报"未缓存"。
            df = _cached_stock_list()
            if df is not None and len(df):
                return len(df), "cache"
            threading.Thread(target=fetch_stock_list, daemon=True,
                             name="health-codelist-warmup").start()
            raise RuntimeError("代码表未缓存（整表 18 页冷启动 6~19s，已转后台预热；"
                               "再调用一次即命中 24h 缓存）")
        return _t

    def _sector_task():
        sec = fetch_sector_list(disp, force=True)
        if not sec:
            raise RuntimeError("slist 返回空")
        return len(sec), None

    def _resolve_task():
        # lookup_name=False：不因查名称而触发 18 页代码表拉取，保证体检在预算内
        r = resolve_symbol(symbol, lookup_name=False)
        if not r.get("code"):
            raise RuntimeError("解析失败")
        return 1, r.get("name")

    tasks: dict[str, Callable[[], Any]] = {
        "daily_sina": _kline_task("sina"),
        "daily_tencent": _kline_task("tencent"),
        "daily_eastmoney": _kline_task("eastmoney"),
        "daily_akshare": _kline_task("akshare"),
        "daily_yfinance": _kline_task("yfinance"),
        "realtime_tencent": _rt_task("tencent"),
        "realtime_sina": _rt_task("sina"),
        "index_sina": _index_task("sina"),
        "index_akshare": _index_task("akshare"),
        "index_tencent": _index_task("tencent"),
        "stock_list_akshare": _list_task("akshare"),
        "stock_list_eastmoney": _list_task("eastmoney"),
        "sector_eastmoney": _sector_task,
        "resolve_symbol": _resolve_task,
    }
    t0 = time.time()

    # ---- HTTP 层原始连通性批量探测（与下面的解析级探测**并行**，不额外占用预算） ----
    tx = _tx_code(info)
    probe_jobs = [
        ("raw_tencent_kline", {"url": TENCENT_KLINE_URL,
                               "params": {"param": "%s,day,,,5,qfq" % tx}}),
        ("raw_sina_kline", {"url": SINA_KLINE_URL,
                            "params": {"symbol": _sina_code(info), "scale": 240,
                                       "ma": "no", "datalen": 5}}),
        ("raw_tencent_quote", {"url": TENCENT_QUOTE_URL + tx}),
        ("raw_sina_quote", {"url": SINA_QUOTE_URL + _sina_code(info),
                            "headers": _SINA_HEADERS}),
        ("raw_em_slist", {"url": EM_SLIST_URL,
                          "params": {"secid": info.get("secid"), "spt": 3, "np": 1,
                                     "pz": 5, "fields": "f12,f14"}}),
        ("raw_em_clist", {"url": EM_CLIST_URL,
                          "params": {"pn": 1, "pz": 5, "np": 1, "fid": "f12",
                                     "fs": "m:0+t:6", "fields": "f12,f14"}}),
        ("raw_em_kline", {"url": EM_KLINE_URL,
                          "params": {"secid": info.get("secid"), "klt": 101, "fqt": 1,
                                     "lmt": 5, "end": 20500101,
                                     "fields1": "f1", "fields2": "f51,f52,f53"}}),
    ]
    raw_box: dict = {}

    def _probe_raw() -> None:
        try:
            raw_box.update(http_get_many(probe_jobs, timeout=8.0, max_workers=8))
        except Exception as exc:  # noqa: BLE001
            log.debug("原始连通性探测失败：%s", exc)

    probe_thread = threading.Thread(target=_probe_raw, daemon=True, name="health-raw")
    probe_thread.start()

    raw = _run_parallel(tasks, timeout)
    items = {name: _health_item(raw.get(name)) for name in tasks}
    probe_thread.join(2.0)
    for name, spec in probe_jobs:
        body = raw_box.get(name)
        ok = bool(body) and len(str(body)) > 2
        items[name] = {"ok": ok, "rows": None, "last_date": None, "ms": None,
                       "error": None if ok else "HTTP 层无响应/被重置"}
    elapsed = round(time.time() - t0, 2)

    daily_keys = ("daily_sina", "daily_tencent", "daily_eastmoney", "daily_akshare",
                  "daily_yfinance")
    rt_keys = ("realtime_tencent", "realtime_sina")
    idx_keys = ("index_sina", "index_akshare", "index_tencent")
    n_ok = sum(1 for it in items.values() if it["ok"])
    daily_ok = any(items[k]["ok"] for k in daily_keys)
    rt_ok = any(items[k]["ok"] for k in rt_keys)
    idx_ok = any(items[k]["ok"] for k in idx_keys)
    if n_ok == 0:
        overall = "offline"
    elif daily_ok and rt_ok and idx_ok and n_ok >= math.ceil(len(items) * 0.5):
        overall = "healthy"
    else:
        overall = "degraded"

    notes: list[str] = []
    ok_list = [k for k, v in items.items() if v["ok"]]
    bad_list = [k for k, v in items.items() if not v["ok"]]
    notes.append("可用源(%d)：%s" % (len(ok_list), ", ".join(ok_list) or "无"))
    notes.append("不可用源(%d)：%s" % (len(bad_list), ", ".join(bad_list) or "无"))
    raw_bad = [k for k in items if k.startswith("raw_") and not items[k]["ok"]]
    if raw_bad:
        notes.append("HTTP 层即不可达的端点：%s（属本机网络/中间设备重置，"
                     "代码已按官方语义实现并自动降级到可用源）" % ", ".join(raw_bad))
    # 每项的底层 HTTP 端点；据此区分"网络不可达"与"可达但解析失败"
    dep = {"daily_sina": ("raw_sina_kline",), "daily_tencent": ("raw_tencent_kline",),
           "daily_eastmoney": ("raw_em_kline",), "daily_akshare": ("raw_em_kline",),
           "index_sina": ("raw_sina_kline",), "index_akshare": ("raw_sina_kline",),
           "index_tencent": ("raw_tencent_kline",),
           "realtime_tencent": ("raw_tencent_quote",),
           "realtime_sina": ("raw_sina_quote",),
           "stock_list_eastmoney": ("raw_em_clist",),
           "sector_eastmoney": ("raw_em_slist",), "stock_list_akshare": (),
           "daily_yfinance": (), "resolve_symbol": ()}
    parse_bad = []
    for k, v in items.items():
        if k.startswith("raw_") or v["ok"]:
            continue
        deps = dep.get(k, ())
        if deps and all(items.get(d, {}).get("ok") for d in deps):
            parse_bad.append(k)       # HTTP 通但取不到值 → 值得排查
    if parse_bad:
        notes.append("⚠ HTTP 层可达但取值为空（需排查解析/参数）：%s" % ", ".join(parse_bad))
    if not daily_ok:
        notes.append("⚠ 无任何日线源可用，K 线相关分析将全部降级")
    if not rt_ok:
        notes.append("⚠ 无任何实时源可用，快照/现价相关因子将缺失")
    if not idx_ok:
        notes.append("⚠ 无任何指数量源可用，大盘对比与 beta 将缺失")
    if not items["daily_yfinance"]["ok"]:
        notes.append("yfinance 不可用（Yahoo 返回 403 / 时区缓存不可写），已作为最后兜底")
    notes.append("代码表项不冷启动拉整表（18 页 6~19s）：已缓存则秒回，未缓存则转后台预热"
                 "并在下一次调用命中；其余各项均 force=True 真实联网测量，"
                 "整体超时上限 %.0fs" % timeout)
    return {"symbol": disp, "name": info.get("name"), "checked_at": _now_str(),
            "elapsed_sec": elapsed, "items": items, "overall": overall, "notes": notes}


# --------------------------------------------------------------------------- #
# 自测（不联网）
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """离线自测：不联网、不发请求，覆盖清洗/解析/空表/合成/缓存。"""
    saved_offline = config.RUNTIME.get("offline", False)
    config.RUNTIME["offline"] = True
    try:
        print("=" * 74)
        print("datasource 离线自测（config.RUNTIME['offline'] = True）")
        print("=" * 74)

        # ---- 1. _sanitize 对坏数据的清洗 ----
        good = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [10.0, 11.0], "high": [11.0, 12.0], "low": [9.5, 10.5],
            "close": [10.5, 11.5], "volume": [1000.0, 2000.0],
            "amount": [np.nan, np.nan], "pct_chg": [np.nan, np.nan],
            "turnover": [np.nan, np.nan]})
        bad = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-06",
                                    "2024-01-07", "2024-01-08", "2024-01-09"]),
            "open": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "high": [11.0, 9.0, 9.0, 12.0, 11.0, 11.0],
            "low": [9.0, 10.0, 9.0, 11.0, 13.0, 9.0],
            #            ^close<=0  ^high<low  ^high<max(o,c)  ^low>min(o,c)  ^high<low  ^close 缺失
            "close": [-1.0, 10.0, 10.0, 10.0, 10.0, np.nan],
            "volume": [1000.0] * 6, "amount": [np.nan] * 6,
            "pct_chg": [np.nan] * 6, "turnover": [np.nan] * 6})
        mixed = pd.concat([good, bad], ignore_index=True)
        clean = _sanitize(mixed)
        assert len(clean) == 2, "清洗后应只剩 2 行好数据，实得 %d" % len(clean)
        assert clean.attrs["dropped"] == 6, clean.attrs
        detail = clean.attrs["dropped_detail"]
        for reason in ("close<=0", "high<low", "high<max(o,c)", "low>min(o,c)",
                       "missing_close"):
            assert reason in detail, "缺少剔除原因统计: %s (%s)" % (reason, detail)
        assert list(clean["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
        print("[1] _sanitize 清洗：输入 %d 行 → 保留 %d 行，剔除 %d 行，原因=%s"
              % (len(mixed), len(clean), clean.attrs["dropped"], detail))

        # ---- 2. 空 DataFrame 的列完整性 ----
        # 说明：offline 的语义是"不发网络请求，只用缓存/合成数据"（见 config.RUNTIME），
        # 所以这里用 force=True 强制绕过缓存，专门验证离线短路分支。
        empty = _empty_kline()
        assert list(empty.columns) == KLINE_COLUMNS
        assert len(empty) == 0
        for fn, kw in ((fetch_daily, {"symbol": "600519"}),
                       (fetch_weekly, {"symbol": "600519"}),
                       (fetch_monthly, {"symbol": "600519"}),
                       (fetch_index, {"symbol": "sh000001"})):
            df = fn(force=True, **kw)
            assert list(df.columns) == KLINE_COLUMNS, "%s 列不齐: %s" % (fn.__name__, list(df.columns))
            assert len(df) == 0, "%s 离线应返回空表" % fn.__name__
            assert df.attrs.get("error"), "%s 空表必须写 attrs['error']" % fn.__name__
        assert fetch_realtime("600519", force=True) == {}, "离线 fetch_realtime 应返回空 dict"
        assert fetch_index_realtime("sh000001", force=True) == {}
        assert fetch_sector_list("600519", force=True) == []
        sl = fetch_stock_list(force=True)
        assert list(sl.columns) == _STOCK_LIST_COLUMNS and len(sl) == 0
        sm = fetch_sector_members("BK0475", force=True)
        assert list(sm.columns) == _SECTOR_MEMBER_COLUMNS and len(sm) == 0
        print("[2] 离线空表：K线列齐全且 attrs['error'] 有值；realtime/板块/代码表均安全返回空")

        # ---- 3. resolve_symbol 各种输入 ----
        cases = ["600519", "sh600519", "600519.SH", "SH600519", "000001", "300750",
                 "00700.HK", "AAPL", "sh000001", "贵州茅台", "茅台", "", None]
        for q in cases:
            r = resolve_symbol(q)
            assert set(r) >= {"code", "name", "display", "prefix", "secid",
                              "yf_symbol", "market", "is_cn", "matched_by"}, r
            print("    resolve_symbol(%-10r) -> code=%-7s prefix=%-3s yf=%-10s by=%s"
                  % (q, r["code"], r["prefix"], r["yf_symbol"], r["matched_by"]))
        assert resolve_symbol("600519")["code"] == "600519"
        assert resolve_symbol("600519")["prefix"] == "sh"
        assert resolve_symbol("000001")["prefix"] == "sz"
        assert resolve_symbol("300750")["prefix"] == "sz"
        assert resolve_symbol("00700.HK")["prefix"] == "hk"
        assert resolve_symbol("AAPL")["market"] == "us"
        assert resolve_symbol("")["matched_by"] == "empty"
        assert resolve_symbol(None)["code"] == ""
        # 代码表可能命中磁盘缓存（offline 的语义是"不发网络请求，只用缓存"），
        # 因此只校验结构契约，不断言一定为空。
        hits = search_symbol("茅台")
        assert isinstance(hits, list)
        for h in hits:
            assert set(h) >= {"code", "name", "display", "prefix", "secid",
                              "yf_symbol", "market", "is_cn", "matched_by"}, h
        print("[3] resolve_symbol：13 种输入全部返回完整结构；"
              "search_symbol 返回结构合法（本次命中 %d 条%s）"
              % (len(hits), "，来自缓存代码表" if hits else "，离线且无缓存"))

        # ---- 4. fetch_daily_synthetic 结构 ----
        syn = fetch_daily_synthetic("600519", days=300, seed=42)
        assert list(syn.columns) == KLINE_COLUMNS
        assert len(syn) == 300, len(syn)
        assert syn.attrs["source"] == "synthetic" and syn.attrs["synthetic"] is True
        assert syn["date"].is_monotonic_increasing
        assert (syn["high"] >= syn[["open", "close"]].max(axis=1) - 1e-9).all()
        assert (syn["low"] <= syn[["open", "close"]].min(axis=1) + 1e-9).all()
        assert (syn["volume"] > 0).all()
        same = fetch_daily_synthetic("600519", days=300, seed=42)
        assert syn["close"].equals(same["close"]), "同 seed 必须可复现"
        print("[4] fetch_daily_synthetic：%d 行 × %d 列，source=synthetic，同 seed 可复现，"
              "日期 %.0f 天，收盘 %.2f→%.2f"
              % (len(syn), len(syn.columns), (syn['date'].iloc[-1] - syn['date'].iloc[0]).days,
                 syn['close'].iloc[0], syn['close'].iloc[-1]))

        # ---- 5. 腾讯行解析（O/C/H/L/V + 分红 dict 第 7 列） ----
        rows = [["2026-10-09", "1255.420", "1263.000", "1282.000", "1255.420", "35111.000",
                 {"nd": "2025", "FHcontent": "10派280.242元"}]]
        tdf = _tencent_rows_to_df(rows, "sh600519")
        assert float(tdf["open"].iloc[0]) == 1255.42 and float(tdf["close"].iloc[0]) == 1263.0
        assert float(tdf["high"].iloc[0]) == 1282.0 and float(tdf["low"].iloc[0]) == 1255.42
        assert float(tdf["volume"].iloc[0]) == 35111.0
        tdf = _tencent_volume_to_shares(tdf, "sh")
        assert float(tdf["volume"].iloc[0]) == 3511100.0, "A股手→股必须 ×100"
        tdf2 = _tencent_volume_to_shares(_tencent_rows_to_df(rows, "hk00700"), "hk")
        assert float(tdf2["volume"].iloc[0]) == 35111.0, "港美股不得 ×100"
        print("[5] 腾讯行解析：O/C/H/L/V 顺序正确，第7列分红 dict 被忽略，"
              "A股量 ×100=3511100 股，港股不换算")

        # ---- 6. 实时字段映射完整性（离线用假报文解析） ----
        fake = ("v_sh600519=\"1~贵州茅台~600519~1263.00~1255.79~1255.42~35111~19932~15179~"
                "1263.00~73~1262.99~2~1262.98~3~1262.88~1~1262.83~8~1263.06~1~1263.11~3~"
                "1263.43~1~1263.44~1~1263.45~3~~20261009161459~7.21~0.57~1282.00~1255.42~"
                "1263.00/35111/4453223189~35111~445322~0.28~19.39~~1282.00~1255.42~2.12~"
                "15788.53~15788.53~6.28~1381.37~1130.21~1.18~78~1268.34~17.73~19.18~~~~0.07~"
                "445322.3189~25.2600~2~   A~GP-A\"")
        body = fake.split('="', 1)[1].rsplit('"', 1)[0]
        parts = body.split("~")
        assert len(parts) >= 40
        assert parts[3] == "1263.00" and parts[TENCENT_FIELD_MAP_KEY("limit_up")] == "1381.37"
        pct = safe_float(parts[TENCENT_FIELD_MAP_KEY("pct_chg")])
        assert abs(pct - 0.57) < 1e-9
        assert TENCENT_FIELD_MAP[38][0] == "turnover" and TENCENT_FIELD_MAP[51][0] == "avg_price"
        assert TENCENT_FIELD_MAP[50][0] == "bid_ask_diff_lots"
        print("[6] TENCENT_FIELD_MAP：%d 个已确认字段（现价[3]/昨收[4]/量[6]/五档[9..28]/"
              "涨跌幅[32]/换手[38]/PE[39]/振幅[43]/市值[44,45]/PB[46]/涨跌停[47,48]/"
              "量比[49]/委差[50]/均价[51]/PE动[52]/PE静[53]/股本[72,73]），"
              "假报文解析 pct_chg=%.2f 涨停价=%s"
              % (len(TENCENT_FIELD_MAP), pct, parts[47]))

        # ---- 7. 缓存读写 / force 语义 ----
        key = cache_key("selftest", "kline")
        cache_put(key, {"v": 1})
        assert cache_get(key, 3600) == {"v": 1}
        assert _cache_read(key, 3600, force=True) is None, "force=True 必须跳过读缓存"
        df_attrs = syn.copy()
        cache_put(key, df_attrs)
        got = cache_get(key, 3600)
        assert isinstance(got, pd.DataFrame)
        assert got.attrs.get("source") == "synthetic", "pickle 往返必须保留 attrs"
        _cache_write_if_ok(cache_key("selftest", "fail"), _empty_kline(), ok=False)
        assert cache_get(cache_key("selftest", "fail"), 3600) is None, "失败结果不得写缓存"
        cache_clear()
        print("[7] 缓存：读写正常、attrs 经 pickle 往返保留、force 短路、失败结果不落盘")

        # ---- 8. 时间/工具函数 ----
        assert _parse_qt_time("20261009161459") == "2026-10-09 16:14:59"
        assert _parse_qt_time("2026/10/09 16:08:14") == "2026-10-09 16:08:14"
        assert _parse_qt_time("") is None
        assert _norm_period("day") == "daily" and _norm_period("W") == "weekly"
        assert _norm_period("monthly") == "monthly" and _norm_period("xxx") == "daily"
        assert _norm_adjust("QFQ") == "qfq" and _norm_adjust("") == "none"
        assert _norm_index_symbol("000300")["display"] == "sh000300"
        assert _norm_index_symbol("000001")["display"] == "sh000001"
        assert _norm_index_symbol("399006")["display"] == "sz399006"
        assert _norm_index_symbol("sh000905")["display"] == "sh000905"
        assert _tx_code(normalize_symbol("700.HK")) == "hk00700"
        assert _tx_code(normalize_symbol("aapl")) == "usAAPL"
        assert _search_key("万  科Ａ") == "万科A"
        print("[8] 工具：时间归一/周期别名/复权别名/指数别名/腾讯代码/中文检索键 全部正确")

        print("-" * 74)
        print("datasource self-test OK")
    finally:
        config.RUNTIME["offline"] = saved_offline


if __name__ == "__main__":  # pragma: no cover
    _self_test()