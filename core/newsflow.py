# -*- coding: utf-8 -*-
"""
消息面与突发事件模块（Newsflow）。

职责
----
把**新闻标题、公司公告、政策口径、黑天鹅事件**压缩成 :class:`core.contracts.Factor`。

产出因子 id
-----------
``news_sentiment_agg``（近 5/10/30 日新闻情感加权）、``announcement_risk``（减持/质押/诉讼/处罚）、
``buyback_incentive``（回购/股权激励/增持）、``restructure_event``、``earnings_preannounce``、
``unlock_pressure``（解禁）、``regulatory_risk``、``geopolitics``、``blackswan``。

设计要点
--------
* **朴素中文情感打分**（:func:`sentiment_score_text`）：利好词 +1、利空词 −1、
  否定词翻转（"不/未/没有/取消/终止/否认"…）、程度副词加权（"大幅/显著/略/小幅"…），
  并专门收录"不及预期/低于预期/下调/由盈转亏/增收不增利"等**复合负面词**；
  最终用 ``tanh`` 归一到 [−1, 1]。
* **reliability 双因子**：来源可信度（公告 0.85 / 官方媒体 0.8 / 普通媒体 0.65 /
  股吧传闻 0.35）× 时效衰减（≤3 天 1.0，≤10 天 0.85，≤30 天 0.7，>30 天 0.5），
  即"是否已被 price in"。

数据源与实测结论
----------------
* 公告：``https://np-anotice-stock.eastmoney.com/api/security/ann`` **实测可用**，
  字段 ``art_code/title/notice_date/columns[].column_name/codes[]``；
  详情页 ``https://data.eastmoney.com/notices/detail/{code}/{art_code}.html``。
* 新闻：东财搜索 ``search-api-web.eastmoney.com/search/jsonp`` 在本机返回
  **与关键词无关的固定内容**（多次不同关键词结果完全一致），因此解析不到
  ``result.cmsArticleWebOld`` 时**自动降级为用公告标题充当新闻**，并通过
  :func:`last_notes` 暴露降级原因；全部失败返回 ``[]``。
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import urllib.parse

import numpy as np
import pandas as pd

from . import config
from .contracts import (Category, DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Factor,
                        clamp, now_cn, safe_float)
from .utils import (cache_key, cached, get_logger, http_get, normalize_symbol,
                    retry_call)

log = get_logger("stockai.newsflow")

__all__ = ["fetch_news", "fetch_announcements", "classify_news", "event_factors",
           "blackswan_scan", "policy_news_scan", "sentiment_score_text",
           "last_notes", "TAG_KEYWORDS", "BULL_WORDS", "BEAR_WORDS",
           "NEGATION_WORDS", "DEGREE_WORDS", "BLACKSWAN_KEYWORDS"]

_EM_ANN = ("https://np-anotice-stock.eastmoney.com/api/security/ann?sr=-1&page_size=30"
           "&page_index=1&ann_type=A&client_source=web&stock_list=%s")
_EM_SEARCH = "https://search-api-web.eastmoney.com/search/jsonp"
_ANN_DETAIL = "https://data.eastmoney.com/notices/detail/%s/%s.html"

# --------------------------------------------------------------------------- #
# 关键词表（模块常量，供调参与人工审阅）
# --------------------------------------------------------------------------- #
#: 新闻标签 → 关键词。**匹配顺序即优先级**（见 :data:`TAG_ORDER`），"其他"兜底。
TAG_KEYWORDS = {
    "突发事件": ["爆炸", "火灾", "事故", "地震", "停产", "召回", "中毒", "坍塌", "泄漏",
                 "被曝", "突发", "紧急"],
    "处罚": ["处罚", "罚款", "警示函", "责令", "没收", "违规", "违法", "造假", "虚增",
             "信息披露违法违规"],
    "监管": ["立案", "调查", "问询", "关注函", "监管函", "稽查", "约谈", "专项检查",
             "反垄断", "制裁", "禁令", "出口管制"],
    "诉讼": ["诉讼", "仲裁", "起诉", "败诉", "判决", "赔偿", "纠纷", "冻结", "查封"],
    "质押": ["质押", "平仓", "强平", "爆仓", "触及平仓线", "补充质押"],
    "减持": ["减持", "清仓", "拟减持", "协议转让", "转让股份", "询价转让"],
    "解禁": ["解禁", "限售股上市", "限售解禁", "解除限售"],
    "重组": ["重组", "并购", "收购", "资产注入", "借壳", "合并", "重大资产", "混改",
             "股权转让"],
    "业绩": ["业绩", "年报", "半年报", "季报", "预增", "预减", "预亏", "预告", "快报",
             "净利润", "营业收入", "扭亏", "增收", "亏损"],
    "订单": ["中标", "订单", "合同", "采购", "框架协议", "签署", "供货", "量产", "投产",
             "获批", "注册证", "提价", "涨价"],
    "回购": ["回购", "注销股份"],
    "增持": ["增持", "举牌", "控股股东增持", "员工持股"],
    "股权激励": ["股权激励", "限制性股票", "期权激励", "激励计划"],
    "分红": ["分红", "派息", "利润分配", "转增", "送股", "股息"],
    "政策": ["政策", "补贴", "减税", "降准", "降息", "专项债", "规划", "指导意见",
             "试点", "以旧换新", "国产替代"],
    "行业": ["行业", "产业链", "供需", "产能", "库存", "价格", "景气", "出货量"],
    "宏观": ["GDP", "CPI", "PPI", "PMI", "社融", "M2", "汇率", "美联储", "加息", "降息",
             "国债", "关税"],
}
#: 标签匹配优先级（越靠前越优先）
TAG_ORDER = ["突发事件", "处罚", "监管", "诉讼", "质押", "减持", "解禁", "重组", "业绩",
             "订单", "回购", "增持", "股权激励", "分红", "政策", "行业", "宏观"]

#: 利好词表（+1）。含**复合词**（整体判正/负，避免被拆分误判）
BULL_WORDS = {
    # 业绩
    "业绩超预期": 1.6, "超预期": 1.3, "预增": 1.2, "大幅增长": 1.3, "同比增长": 0.8,
    "扭亏为盈": 1.5, "扭亏": 1.2, "创历史新高": 1.4, "创新高": 1.1, "高增长": 1.1,
    "盈利": 0.8, "增长": 0.6, "回升": 0.7, "改善": 0.7, "翻倍": 1.3, "上调": 1.0,
    "超市场预期": 1.5, "减亏": 0.4,
    # 订单/经营
    "中标": 1.2, "订单": 0.9, "大额订单": 1.4, "签署": 0.7, "合同": 0.6, "合作": 0.7,
    "战略合作": 1.0, "量产": 1.0, "投产": 0.9, "扩产": 0.9, "满产": 1.0, "提价": 1.1,
    "涨价": 1.0, "获批": 1.0, "通过审批": 1.1, "注册批件": 1.1, "专利": 0.7,
    "突破": 0.9, "首家": 0.8, "龙头": 0.7, "放量": 0.8,
    # 资本运作
    "回购": 1.1, "增持": 1.1, "举牌": 1.2, "股权激励": 1.0, "员工持股": 0.9,
    "分红": 0.8, "高分红": 1.1, "派息": 0.7, "重组": 1.0, "并购": 0.9, "资产注入": 1.2,
    "借壳": 1.2, "引入战投": 1.0, "混改": 0.9,
    # 政策/宏观
    "补贴": 1.0, "减税": 1.0, "税收优惠": 1.0, "专项债": 0.9, "以旧换新": 0.9,
    "国产替代": 1.1, "政策支持": 1.1, "试点": 0.7, "降准": 1.0, "降息": 1.0,
    "景气": 0.9, "复苏": 0.9, "涨价预期": 0.9,
}

#: 利空词表（−1）。复合负面词（"不及预期""增收不增利"…）**必须整体收录**
BEAR_WORDS = {
    # 业绩
    "不及预期": 1.5, "低于预期": 1.5, "未达预期": 1.4, "逊于预期": 1.3, "预减": 1.2,
    "预亏": 1.4, "亏损扩大": 1.5, "由盈转亏": 1.6, "增收不增利": 1.4, "业绩变脸": 1.6,
    "下滑": 1.0, "下降": 0.8, "同比下降": 0.9, "大幅下滑": 1.4, "暴跌": 1.5, "腰斩": 1.4,
    "下调": 1.1, "下调评级": 1.3, "下调业绩": 1.4, "亏损": 1.1, "承压": 0.9, "疲软": 0.9,
    # 资本运作/股东
    "减持": 1.2, "清仓": 1.3, "拟减持": 1.2, "套现": 1.1, "质押": 0.9, "爆仓": 1.6,
    "平仓": 1.3, "冻结": 1.2, "解禁": 0.9, "限售解禁": 1.0, "终止": 1.0, "失败": 1.0,
    "终止重组": 1.4, "取消": 0.8, "撤回": 0.8,
    # 监管/合规
    "立案": 1.6, "调查": 1.2, "处罚": 1.3, "罚款": 1.0, "违规": 1.2, "违法": 1.4,
    "造假": 1.8, "财务造假": 2.0, "虚增": 1.6, "警示函": 1.0, "问询": 0.8, "诉讼": 1.0,
    "仲裁": 0.9, "败诉": 1.2, "赔偿": 1.0, "退市": 1.8, "非标": 1.4, "保留意见": 1.3,
    "商誉减值": 1.4, "计提": 0.9, "减值": 1.1, "债务违约": 1.9, "逾期": 1.2,
    "无法兑付": 1.8, "失联": 1.6, "被留置": 1.5, "被带走": 1.5, "协助调查": 1.3,
    # 行业/宏观
    "降价": 1.0, "价格战": 1.2, "内卷": 1.0, "产能过剩": 1.2, "限产": 0.9, "停产": 1.2,
    "召回": 1.2, "制裁": 1.4, "反倾销": 1.1, "禁令": 1.2, "关税": 0.9, "断供": 1.4,
    "出口管制": 1.3, "衰退": 1.2,
}

#: 否定词（出现于情感词之前 4 字内则翻转语义）
NEGATION_WORDS = ["不", "未", "没有", "无", "非", "难以", "尚未", "并未", "否认", "不会",
                  "不再", "免于", "避免", "取消", "终止", "停止", "失败"]
#: 程度副词 → 权重倍数（取命中词的最大值）
DEGREE_WORDS = {"大幅": 1.4, "显著": 1.3, "明显": 1.2, "超": 1.2, "巨额": 1.5,
                "严重": 1.5, "急剧": 1.4, "暴": 1.5, "创纪录": 1.4, "略": 0.6,
                "小幅": 0.7, "轻微": 0.6, "微": 0.6, "有所": 0.8, "持续": 1.1}

#: 黑天鹅类型 → (关键词, 默认严重度)
BLACKSWAN_KEYWORDS = {
    "财务造假/立案调查": (["财务造假", "虚增", "造假", "立案调查", "被立案", "涉嫌信息披露",
                          "违法违规", "欺诈发行"], 0.95),
    "大股东爆仓": (["爆仓", "强制平仓", "触及平仓线", "被强平", "质押违约"], 0.85),
    "债务违约": (["债务违约", "债券违约", "无法兑付", "逾期未付", "实质性违约", "评级下调"], 0.85),
    "实控人风险": (["失联", "被留置", "被带走", "协助调查", "被监视居住", "董事长辞职",
                    "高管被查"], 0.8),
    "退市风险": (["退市风险", "面值退市", "终止上市", "ST", "暂停上市", "退市整理"], 0.9),
    "审计非标": (["非标意见", "保留意见", "无法表示意见", "否定意见", "审计机构辞任"], 0.75),
    "重大诉讼": (["重大诉讼", "诉讼", "仲裁", "败诉", "巨额赔偿", "被起诉"], 0.6),
    "重大事故": (["爆炸", "火灾", "坍塌", "泄漏", "重大安全事故", "人员伤亡", "停产整顿"], 0.85),
    "监管重罚": (["行政处罚", "市场禁入", "巨额罚款", "责令停产", "吊销", "制裁清单"], 0.7),
}

#: 地缘政治关键词（归入 geopolitics 因子）
GEOPOLITICS_WORDS = ["制裁", "关税", "贸易战", "出口管制", "实体清单", "禁令", "断供",
                     "冲突", "战争", "地缘", "台海", "南海", "脱钩", "封锁", "反倾销"]

#: 来源可信度（用于 reliability）
SOURCE_CREDIBILITY = {
    "公告": 0.85, "交易所": 0.85,
    "证券时报": 0.8, "上海证券报": 0.8, "中国证券报": 0.8, "证券日报": 0.8,
    "新华社": 0.8, "央视": 0.8, "人民日报": 0.8,
    "东方财富": 0.7, "e公司": 0.7, "财联社": 0.7, "第一财经": 0.7, "界面": 0.65,
    "股吧": 0.35, "传闻": 0.3, "雪球": 0.4,
}
DEFAULT_CREDIBILITY = 0.65

_LAST_NOTES: list = []


def last_notes() -> list:
    """最近一次取数的降级说明（中文）。"""
    return list(_LAST_NOTES)


def _note(msg: str) -> None:
    if msg and msg not in _LAST_NOTES:
        _LAST_NOTES.append(msg)
    log.debug(msg)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _num(x, default=float("nan")) -> float:
    return safe_float(x, default)


def _code(symbol) -> str:
    try:
        return normalize_symbol(symbol)["code"]
    except Exception:  # noqa: BLE001
        return re.sub(r"\D", "", str(symbol or ""))[:6]


def _today() -> dt.date:
    return now_cn().date()


def _parse_date(s) -> dt.date | None:
    """尽力解析日期字符串（``2026-08-15 00:00:00`` / ``2026-08-15`` / 时间戳）。"""
    if s is None:
        return None
    if isinstance(s, (dt.date, dt.datetime)):
        return s.date() if isinstance(s, dt.datetime) else s
    txt = str(s).strip()[:19]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(txt[:len(fmt) + 2].strip(), fmt).date()
        except Exception:  # noqa: BLE001
            continue
    try:
        return pd.to_datetime(txt, errors="coerce").date()
    except Exception:  # noqa: BLE001
        return None


def _days_ago(date_str) -> float:
    d = _parse_date(date_str)
    if d is None:
        return float("nan")
    return float((_today() - d).days)


def _clean(text: str) -> str:
    """清掉 ``<em>`` 等标签与多余空白。"""
    t = re.sub(r"<[^>]+>", "", str(text or ""))
    return re.sub(r"\s+", " ", t).strip()


def _credibility(source: str) -> float:
    s = str(source or "")
    for k, v in SOURCE_CREDIBILITY.items():
        if k in s:
            return v
    return DEFAULT_CREDIBILITY


def _recency_mult(days: float) -> float:
    """时效衰减：越旧越可能已被 price in。"""
    if days is None or (isinstance(days, float) and math.isnan(days)):
        return 0.7
    if days <= 3:
        return 1.0
    if days <= 10:
        return 0.85
    if days <= 30:
        return 0.7
    return 0.5


def _reliability(items: list, base: float = 0.7) -> float:
    """新闻类因子可信度 = 来源可信度均值 × 最新一条的时效衰减。"""
    if not items:
        return clamp(base * 0.3, 0.05, 0.5)
    creds = [_credibility(it.get("source")) for it in items]
    days = [_days_ago(it.get("date")) for it in items]
    days = [d for d in days if not math.isnan(d)]
    rec = _recency_mult(min(days)) if days else 0.7
    return clamp(float(np.mean(creds)) * rec, 0.05, 0.95)


def _neutral(fid: str, name: str, detail: str, reliability: float = 0.2, value=None,
             theory: str = "事件驱动", horizon=("d1", "d5", "d10", "d20")) -> Factor:
    return Factor(fid, name, Category.NEWS, DIR_NEUTRAL, 0.0,
                  clamp(reliability, 0.05, 0.5), value=value, detail=detail,
                  theory=theory, horizon=horizon)


def _amount_yuan(text: str) -> float:
    """从文本里抽金额（元）：支持 "1.2亿元" "5000万元" "3,000万"。"""
    m = re.search(r"([0-9][0-9,\.]*)\s*(亿元|亿|万元|万)", str(text or ""))
    if not m:
        return float("nan")
    try:
        v = float(m.group(1).replace(",", ""))
    except ValueError:
        return float("nan")
    unit = m.group(2)
    mult = {"亿元": 1e8, "亿": 1e8, "万元": 1e4, "万": 1e4}[unit]
    return v * mult


# --------------------------------------------------------------------------- #
# 1. 情感打分
# --------------------------------------------------------------------------- #
def sentiment_score_text(text) -> float:
    """朴素中文情感打分，返回 −1~1。

    规则
    ----
    1. 词典命中：``BULL_WORDS`` 记 +权重，``BEAR_WORDS`` 记 −权重（**按词长降序匹配**，
       保证"不及预期""增收不增利"等复合词优先于内部的"预期""增长"）。
    2. 否定翻转：情感词前 4 字内出现 ``NEGATION_WORDS`` → 符号翻转并 ×0.8
       （如"未亏损"由负转正）。
    3. 程度加权：情感词前 3 字内出现 ``DEGREE_WORDS`` → 乘以对应倍数
       （"大幅预增" ×1.4，"略增" ×0.6）。
    4. 归一化：``tanh(总分 / 2)`` → [−1, 1]；空文本返回 0.0。

    Examples
    --------
    >>> round(sentiment_score_text("公司业绩大幅预增，净利润同比增长80%"), 2) > 0
    True
    >>> round(sentiment_score_text("业绩不及预期，净利润大幅下滑"), 2) < 0
    True
    """
    try:
        t = _clean(text)
        if not t:
            return 0.0
        words = sorted(list(BULL_WORDS.items()) + [(w, -s) for w, s in BEAR_WORDS.items()],
                       key=lambda kv: -len(kv[0]))
        raw = 0.0
        used = []          # 已占用的字符区间，避免复合词与内部词重复计分
        for w, s in words:
            start = 0
            while True:
                i = t.find(w, start)
                if i < 0:
                    break
                span = (i, i + len(w))
                if any(not (span[1] <= a or span[0] >= b) for a, b in used):
                    start = i + 1
                    continue
                used.append(span)
                score = float(s)
                pre4 = t[max(0, i - 4):i]
                pre3 = t[max(0, i - 3):i]
                if any(neg in pre4 for neg in NEGATION_WORDS):
                    score = -score * 0.8
                mult = 1.0
                for d, m in DEGREE_WORDS.items():
                    if d in pre3:
                        mult = max(mult, m) if m >= 1.0 else min(mult, m)
                raw += score * mult
                start = i + len(w)
        return float(math.tanh(raw / 2.0))
    except Exception as exc:  # noqa: BLE001
        log.debug("情感打分异常: %r", exc)
        return 0.0


def _match_tag(title: str) -> str:
    """按 :data:`TAG_ORDER` 优先级给标题打标签。"""
    t = str(title or "")
    for tag in TAG_ORDER:
        for kw in TAG_KEYWORDS.get(tag, []):
            if kw in t:
                return tag
    return "其他"


#: 各标签的默认方向先验（业绩/订单/重组等由情感打分决定）
TAG_DIRECTION = {"减持": -1, "质押": -1, "解禁": -1, "诉讼": -1, "处罚": -1, "监管": -1,
                 "回购": 1, "增持": 1, "股权激励": 1, "分红": 1, "突发事件": -1}


def classify_news(items) -> list:
    """给每条新闻/公告打标签与情感分。

    Returns
    -------
    list[dict]
        ``[{"title","date","source","url","tag","direction","strength","detail"}]``，
        ``direction ∈ {+1,−1,0}``，``strength ∈ [0,1]``。
    """
    out = []
    for it in items or []:
        try:
            if not isinstance(it, dict):
                continue
            title = _clean(it.get("title") or it.get("title_ch") or "")
            if not title:
                continue
            tag = _match_tag(title)
            senti = sentiment_score_text(title)
            prior = TAG_DIRECTION.get(tag, 0)
            if abs(senti) >= 0.08:
                direction = 1 if senti > 0 else -1
            else:
                direction = prior
            strength = clamp(max(abs(senti), 0.25 if direction else 0.0), 0, 1)
            if tag in ("处罚", "监管", "诉讼", "质押", "减持", "解禁", "突发事件"):
                strength = clamp(max(strength, 0.45), 0, 1)
            out.append({
                "title": title, "date": it.get("date") or it.get("notice_date") or "",
                "source": it.get("source") or "未知来源", "url": it.get("url") or "",
                "tag": tag, "direction": direction, "strength": round(float(strength), 3),
                "sentiment": round(float(senti), 3),
                "detail": "%s类消息，情感分 %+.2f（%s）"
                          % (tag, senti, "利好" if direction > 0 else
                             ("利空" if direction < 0 else "中性")),
            })
        except Exception as exc:  # noqa: BLE001
            log.debug("classify_news 单条异常: %r", exc)
    return out


# --------------------------------------------------------------------------- #
# 2. 取数
# --------------------------------------------------------------------------- #
def _producer_announcements(symbol) -> list:
    """东财公告（实测可用）。

    返回 ``[{"title","date","source","url","type","code","art_code"}]``，按日期降序。
    """
    code = _code(symbol)
    payload = retry_call(lambda: http_get(_EM_ANN % code, json_body=True), attempts=3,
                         delay=0.8, default=None, label="announcements")
    rows = ((payload or {}).get("data") or {}).get("list") or []
    out = []
    for r in rows:
        art = str(r.get("art_code") or "")
        cols = r.get("columns") or []
        cname = (cols[0].get("column_name") if cols and isinstance(cols[0], dict) else "")
        out.append({
            "title": _clean(r.get("title") or r.get("title_ch") or ""),
            "date": str(r.get("notice_date") or r.get("display_time") or "")[:10],
            "source": "公告", "type": cname or "公告",
            "url": _ANN_DETAIL % (code, art) if art else "",
            "code": code, "art_code": art,
        })
    out = [x for x in out if x["title"]]
    out.sort(key=lambda x: str(x.get("date") or ""), reverse=True)
    if out:
        _note("公告来源：东财 np-anotice（%d 条，最新 %s）" % (len(out), out[0]["date"]))
    else:
        _note("东财公告接口无数据（或请求失败）")
    return out


def fetch_announcements(symbol, force=False) -> list:
    """公司公告列表（按日期降序）；失败返回 ``[]``。"""
    code = _code(symbol)
    try:
        if force:
            return _producer_announcements(symbol)
        out = cached(cache_key("ann", code), config.RUNTIME["cache_ttl_daily"],
                     lambda: _producer_announcements(symbol))
        return out if isinstance(out, list) else []
    except Exception as exc:  # noqa: BLE001
        _note("fetch_announcements 异常降级：%r" % exc)
        return []


def _em_search(keyword: str, page_size: int = 20) -> list:
    """东财搜索接口（本机返回固定内容 → 解析不到时返回 []）。"""
    param = {
        "uid": "", "keyword": keyword, "type": ["cmsArticleWebOld"], "client": "web",
        "clientType": "web", "clientVersion": "curr",
        "param": {"cmsArticleWebOld": {"searchScope": "default", "sort": "time",
                                       "pageIndex": 1, "pageSize": page_size,
                                       "preTag": "<em>", "postTag": "</em>"}},
    }
    url = _EM_SEARCH + "?cb=cb&param=" + urllib.parse.quote(
        json.dumps(param, ensure_ascii=False))
    payload = retry_call(lambda: http_get(url, json_body=True), attempts=2, delay=0.6,
                         default=None, label="em_search")
    if not isinstance(payload, dict):
        return []
    res = payload.get("result") or {}
    arts = res.get("cmsArticleWebOld") or []
    out = []
    for a in arts:
        out.append({
            "title": _clean(a.get("title")),
            "date": str(a.get("date") or "")[:10],
            "source": _clean(a.get("mediaName") or a.get("source") or "东方财富"),
            "url": a.get("url") or "",
            "summary": _clean(a.get("content") or "")[:200],
        })
    return [x for x in out if x["title"]]


def fetch_news(symbol, name="", days=30, force=False) -> list:
    """个股新闻（降级链：东财搜索 → 公告标题充当新闻 → ``[]``）。

    Returns
    -------
    list[dict]
        ``[{"title","date","source","url","summary"}]``（按日期降序，仅保留最近
        ``days`` 天；日期解析失败的一律保留）。拿不到时返回 ``[]``，
        降级原因可通过 :func:`last_notes` 查看。
    """
    code = _code(symbol)
    key = cache_key("news", code, int(days))

    def _p() -> list:
        kw = str(name or code)
        items = []
        try:
            items = _em_search(kw)
            if items:
                _note("新闻来源：东财搜索接口（%d 条）" % len(items))
        except Exception as exc:  # noqa: BLE001
            _note("东财搜索接口失败：%r" % exc)
        if not items:
            _note("东财搜索接口未返回可用新闻（本机实测返回与关键词无关的固定响应），"
                  "降级用公告标题充当新闻")
            anns = fetch_announcements(symbol)
            items = [{"title": a.get("title"), "date": a.get("date"),
                      "source": "公告", "url": a.get("url"),
                      "summary": a.get("type") or "公告"} for a in anns]
        # 时间过滤
        limit = max(1, int(days))
        keep = []
        for it in items:
            d = _days_ago(it.get("date"))
            if math.isnan(d) or d <= limit:
                keep.append(it)
        keep.sort(key=lambda x: str(x.get("date") or ""), reverse=True)
        if not keep:
            _note("最近 %d 天内没有可用新闻/公告" % limit)
        return keep

    try:
        if force:
            return _p()
        out = cached(key, config.RUNTIME["cache_ttl_seconds"], _p)
        return out if isinstance(out, list) else []
    except Exception as exc:  # noqa: BLE001
        _note("fetch_news 异常降级：%r" % exc)
        return []


# --------------------------------------------------------------------------- #
# 3. 黑天鹅与政策扫描
# --------------------------------------------------------------------------- #
def blackswan_scan(news, frame=None) -> dict:
    """黑天鹅事件扫描。

    识别：财务造假/立案调查、大股东爆仓、债务违约、实控人风险（失联/被留置）、
    退市风险、审计非标、重大诉讼（**金额占净资产比过高**）、重大事故、监管重罚。
    ``frame`` 提供财务表时可计算"诉讼金额 / 净资产"或"诉讼金额 / 净利润"比例。

    Returns
    -------
    dict
        ``{"flags":[{"type","title","severity"}...],"severity":0~1,"detail"}``。
    """
    out = {"flags": [], "severity": 0.0, "detail": "未发现黑天鹅信号"}
    try:
        items = news if isinstance(news, (list, tuple)) else []
        flags = []
        for it in items:
            if not isinstance(it, dict):
                continue
            title = _clean(it.get("title") or "")
            if not title:
                continue
            for typ, (kws, sev) in BLACKSWAN_KEYWORDS.items():
                hit = [k for k in kws if k in title]
                if not hit:
                    continue
                severity = float(sev)
                extra = ""
                if typ == "重大诉讼":
                    amt = _amount_yuan(title)
                    net_asset = float("nan")
                    net_profit = float("nan")
                    if isinstance(frame, pd.DataFrame) and not frame.empty:
                        for col in ("净资产", "股东权益合计(净资产)", "所有者权益",
                                    "股东权益"):
                            if col in frame.columns:
                                try:
                                    net_asset = float(pd.to_numeric(
                                        frame[col], errors="coerce").dropna().iloc[-1])
                                    break
                                except Exception:  # noqa: BLE001
                                    pass
                        if "净利润" in frame.columns:
                            try:
                                net_profit = float(pd.to_numeric(
                                    frame["净利润"], errors="coerce").dropna().iloc[-1])
                            except Exception:  # noqa: BLE001
                                net_profit = float("nan")
                    if not math.isnan(amt):
                        if not math.isnan(net_asset) and net_asset > 0:
                            ratio = amt / net_asset * 100.0
                            extra = "；涉案金额 %.2f 亿元，占净资产 %.1f%%" % (amt / 1e8, ratio)
                            if ratio > 10:
                                severity = min(0.99, severity + 0.3)
                            elif ratio > 3:
                                severity = min(0.99, severity + 0.15)
                        elif not math.isnan(net_profit) and net_profit > 0 and amt > net_profit:
                            extra = "；涉案金额 %.2f 亿元已超最近一期净利润" % (amt / 1e8)
                            severity = min(0.99, severity + 0.2)
                        else:
                            extra = "；涉案金额 %.2f 亿元（缺少净资产数据，按绝对值评估）" \
                                    % (amt / 1e8)
                            severity = min(0.99, severity + (0.2 if amt > 5e8 else 0.0))
                flags.append({"type": typ, "title": title,
                              "severity": round(float(severity), 3),
                              "matched": hit, "date": it.get("date") or "",
                              "detail": ("命中关键词：%s%s" % ("、".join(hit[:4]), extra))})
        # 同一类型去重（保留最高严重度），并按严重度排序
        best: dict = {}
        for f in flags:
            k = f["type"]
            if k not in best or f["severity"] > best[k]["severity"]:
                best[k] = f
        out["flags"] = sorted(best.values(), key=lambda x: -x["severity"])
        if out["flags"]:
            sev = out["flags"][0]["severity"]
            if len(out["flags"]) > 1:      # 多类风险并发 → 适当加权
                sev = min(1.0, sev * 0.75 + float(np.mean(
                    [f["severity"] for f in out["flags"][1:]])) * 0.25 + 0.05)
            out["severity"] = round(float(clamp(sev, 0, 1)), 3)
            out["detail"] = "检出 %d 类风险事件：%s（最高严重度 %.2f）" % (
                len(out["flags"]),
                "；".join("%s(%s)" % (f["type"], f["title"][:24]) for f in out["flags"][:3]),
                out["severity"])
    except Exception as exc:  # noqa: BLE001
        out["detail"] = "黑天鹅扫描异常，已降级：%r" % exc
        _note(out["detail"])
    return out


def policy_news_scan(items) -> dict:
    """政策类新闻扫描。

    Returns
    -------
    dict
        ``{"policy_score":-1~1,"themes":[...],"items":[{"title","keyword","direction"}]}``。
    """
    out = {"policy_score": 0.0, "themes": [], "items": [], "detail": ""}
    try:
        good = config.POLICY_KEYWORDS.get("利好", [])
        bad = config.POLICY_KEYWORDS.get("利空", [])
        hits, score = [], 0.0
        texts = []
        for it in items or []:
            if not isinstance(it, dict):
                continue
            title = _clean(it.get("title") or "")
            if not title:
                continue
            texts.append(title)
            for kw in good:
                if kw in title:
                    score += 1.0
                    hits.append({"title": title, "keyword": kw, "direction": 1})
            for kw in bad:
                if kw in title:
                    score -= 1.0
                    hits.append({"title": title, "keyword": kw, "direction": -1})
        blob = " ".join(texts)
        themes = [t for t in config.HOT_THEMES if t and t in blob]
        out.update({
            "policy_score": round(float(math.tanh(score / 3.0)), 3),
            "themes": themes, "items": hits[:20],
            "detail": ("扫描 %d 条标题，命中政策关键词 %d 次（利好 %d / 利空 %d），"
                       "题材：%s" % (len(texts), len(hits),
                                   sum(1 for h in hits if h["direction"] > 0),
                                   sum(1 for h in hits if h["direction"] < 0),
                                   "、".join(themes) if themes else "无")),
        })
    except Exception as exc:  # noqa: BLE001
        out["detail"] = "政策扫描异常，已降级：%r" % exc
        _note(out["detail"])
    return out


# --------------------------------------------------------------------------- #
# 4. 因子合成
# --------------------------------------------------------------------------- #
_EVENT_H = ("d1", "d5", "d10", "d20")
_LONG_EVENT_H = ("d1", "d5", "d10", "d20", "q1", "h1")


def _agg_sentiment(items: list) -> tuple:
    """近 5/10/30 日情感加权（权重 0.5/0.3/0.2），返回 ``(score, detail)``。"""
    buckets = {5: [], 10: [], 30: []}
    for it in items:
        d = _days_ago(it.get("date"))
        if math.isnan(d):
            buckets[5].append(it)      # 无日期按最新处理
            continue
        for w in (5, 10, 30):
            if d <= w:
                buckets[w].append(it)
    scores, parts = [], []
    for w, weight in ((5, 0.5), (10, 0.3), (30, 0.2)):
        b = buckets[w]
        if not b:
            continue
        vals = [sentiment_score_text(x.get("title")) for x in b]
        s = float(np.mean(vals)) if vals else 0.0
        scores.append((s, weight))
        parts.append("近 %d 日 %d 条均分 %+.2f" % (w, len(b), s))
    wsum = sum(w for _, w in scores)
    score = sum(s * w for s, w in scores) / wsum if wsum else 0.0
    return score, "；".join(parts) if parts else "无可用新闻"


def event_factors(symbol, news=None, announcements=None, name="") -> list:
    """产出消息面/事件类因子列表。

    Parameters
    ----------
    symbol : str
        股票代码。
    news : list | None
        :func:`fetch_news` 的产出（``[{"title","date","source"}...]``）。
    announcements : list | None
        :func:`fetch_announcements` 的产出。
    name : str
        股票名称（仅用于说明文案）。

    Returns
    -------
    list[Factor]
        id 见模块 docstring。``news`` 与 ``announcements`` 都为空时返回一批
        中性 + 低可信度因子（**不抛异常**）。
    """
    fs: list = []
    try:
        nw = [x for x in (news or []) if isinstance(x, dict)]
        an = [x for x in (announcements or []) if isinstance(x, dict)]
        all_items = nw + [dict(a, source=a.get("source") or "公告") for a in an]
        cls = classify_news(all_items)
        tags = {}
        for c in cls:
            tags.setdefault(c["tag"], []).append(c)

        # ---- 1) 新闻情感加权 --------------------------------------------- #
        score, detail = _agg_sentiment(all_items)
        if not all_items:
            fs.append(_neutral("news_sentiment_agg", "新闻情感", "没有可用新闻/公告数据",
                               0.2, theory="消息面驱动"))
        else:
            fs.append(Factor("news_sentiment_agg", "新闻情感", Category.NEWS,
                             DIR_BULL if score > 0.05 else (DIR_BEAR if score < -0.05 else 0),
                             clamp(abs(score) * 2.0, 0, 1), _reliability(all_items, 0.75),
                             value=round(float(score), 3),
                             detail="新闻情感加权分 %+.2f（%s）" % (score, detail),
                             theory="消息面驱动 / 情感分析", horizon=_EVENT_H))

        # ---- 2) 公告风险（减持/质押/诉讼/处罚） -------------------------- #
        risk_tags = ("减持", "质押", "诉讼", "处罚", "监管")
        risk_items = [c for c in cls if c["tag"] in risk_tags]
        if not risk_items:
            fs.append(_neutral("announcement_risk", "公告风险",
                               "最近公告中未发现减持/质押/诉讼/处罚类事项", 0.5,
                               theory="事件驱动 / 风险预警", horizon=_EVENT_H))
        else:
            sev = float(np.mean([c["strength"] for c in risk_items]))
            n_recent = sum(1 for c in risk_items if _num(_days_ago(c["date"]), 999) <= 30)
            strength = clamp(sev * 0.7 + clamp(n_recent / 5.0, 0, 1) * 0.3, 0.2, 0.95)
            fs.append(Factor("announcement_risk", "公告风险", Category.NEWS, DIR_BEAR,
                             strength, _reliability(risk_items, 0.8),
                             value="%d 条" % len(risk_items),
                             detail="检出 %d 条风险类公告（%s）：%s"
                                    % (len(risk_items),
                                       "、".join(sorted({c["tag"] for c in risk_items})),
                                       "；".join(c["title"][:26] for c in risk_items[:2])),
                             theory="事件驱动 / 风险预警", horizon=_EVENT_H))

        # ---- 3) 回购/增持/股权激励 --------------------------------------- #
        good_items = [c for c in cls if c["tag"] in ("回购", "增持", "股权激励")]
        if not good_items:
            fs.append(_neutral("buyback_incentive", "回购增持激励",
                               "最近公告中未发现回购/增持/股权激励", 0.45,
                               theory="信号理论（内部人买入）", horizon=_EVENT_H))
        else:
            strength = clamp(0.4 + 0.15 * len(good_items), 0.2, 0.9)
            fs.append(Factor("buyback_incentive", "回购增持激励", Category.NEWS, DIR_BULL,
                             strength, _reliability(good_items, 0.8),
                             value="%d 条" % len(good_items),
                             detail="%s（%d 条）：%s"
                                    % ("、".join(sorted({c["tag"] for c in good_items})),
                                       len(good_items),
                                       "；".join(c["title"][:26] for c in good_items[:2])),
                             theory="信号理论（内部人买入）", horizon=_EVENT_H))

        # ---- 4) 重组事件 -------------------------------------------------- #
        re_items = [c for c in cls if c["tag"] == "重组"]
        if not re_items:
            fs.append(_neutral("restructure_event", "重组事件", "近期无重组/并购类公告", 0.3,
                               theory="事件驱动 / 重组预期",
                               horizon=("q1", "h1")))
        else:
            senti = float(np.mean([c["sentiment"] for c in re_items]))
            fs.append(Factor("restructure_event", "重组事件", Category.NEWS,
                             DIR_BULL if senti >= 0 else DIR_BEAR,
                             clamp(0.5 + abs(senti), 0.3, 0.9),
                             _reliability(re_items, 0.75),
                             value="%d 条" % len(re_items),
                             detail="重组/并购类消息 %d 条（情感 %+.2f）：%s"
                                    % (len(re_items), senti,
                                       "；".join(c["title"][:30] for c in re_items[:2])),
                             theory="事件驱动 / 重组预期",
                             horizon=("d1", "d5", "d10", "d20", "q1", "h1")))

        # ---- 5) 业绩预告 -------------------------------------------------- #
        ea_items = [c for c in cls
                    if c["tag"] == "业绩" and any(k in c["title"]
                                                  for k in ("预告", "快报", "预增", "预减",
                                                            "预亏", "业绩", "年报", "季报",
                                                            "半年报"))]
        if not ea_items:
            fs.append(_neutral("earnings_preannounce", "业绩预告",
                               "近期无业绩预告/快报类公告", 0.3,
                               theory="盈余惊喜", horizon=("d1", "d5", "q1", "h1")))
        else:
            senti = float(np.mean([c["sentiment"] for c in ea_items]))
            fs.append(Factor("earnings_preannounce", "业绩预告", Category.NEWS,
                             DIR_BULL if senti > 0.05 else (DIR_BEAR if senti < -0.05 else 0),
                             clamp(abs(senti) * 1.5, 0, 1), _reliability(ea_items, 0.8),
                             value="%d 条" % len(ea_items),
                             detail="业绩类公告 %d 条，情感 %+.2f：%s"
                                    % (len(ea_items), senti,
                                       "；".join(c["title"][:28] for c in ea_items[:2])),
                             theory="盈余惊喜 / 业绩预告效应",
                             horizon=("d1", "d5", "d10", "q1", "h1")))

        # ---- 6) 解禁压力 -------------------------------------------------- #
        un_items = [c for c in cls if c["tag"] == "解禁"]
        if not un_items:
            fs.append(_neutral("unlock_pressure", "解禁压力", "近期无解禁公告", 0.4,
                               theory="供给冲击", horizon=_EVENT_H))
        else:
            fs.append(Factor("unlock_pressure", "解禁压力", Category.NEWS, DIR_BEAR,
                             clamp(0.4 + 0.15 * len(un_items), 0.2, 0.9),
                             _reliability(un_items, 0.8),
                             value="%d 条" % len(un_items),
                             detail="解禁/限售上市公告 %d 条：%s"
                                    % (len(un_items),
                                       "；".join(c["title"][:28] for c in un_items[:2])),
                             theory="供给冲击 / 限售解禁", horizon=_EVENT_H))

        # ---- 7) 监管风险 -------------------------------------------------- #
        reg_items = [c for c in cls if c["tag"] in ("处罚", "监管", "诉讼")]
        if not reg_items:
            fs.append(_neutral("regulatory_risk", "监管风险", "近期无监管/处罚类消息", 0.45,
                               theory="政策风险溢价", horizon=_EVENT_H))
        else:
            fs.append(Factor("regulatory_risk", "监管风险", Category.NEWS, DIR_BEAR,
                             clamp(0.5 + 0.1 * len(reg_items), 0.3, 0.95),
                             _reliability(reg_items, 0.8),
                             value="%d 条" % len(reg_items),
                             detail="监管/处罚/诉讼类消息 %d 条：%s"
                                    % (len(reg_items),
                                       "；".join(c["title"][:28] for c in reg_items[:2])),
                             theory="政策风险溢价", horizon=_EVENT_H))

        # ---- 8) 地缘政治 -------------------------------------------------- #
        geo_items = [c for c in cls if any(k in c["title"] for k in GEOPOLITICS_WORDS)]
        if not geo_items:
            fs.append(_neutral("geopolitics", "地缘政治", "近期无地缘政治相关消息", 0.4,
                               theory="地缘风险溢价", horizon=_EVENT_H))
        else:
            fs.append(Factor("geopolitics", "地缘政治", Category.NEWS, DIR_BEAR,
                             clamp(0.35 + 0.15 * len(geo_items), 0.2, 0.85),
                             _reliability(geo_items, 0.6),
                             value="%d 条" % len(geo_items),
                             detail="地缘政治类消息 %d 条：%s"
                                    % (len(geo_items),
                                       "；".join(c["title"][:28] for c in geo_items[:2])),
                             theory="地缘风险溢价", horizon=_EVENT_H))

        # ---- 9) 黑天鹅 ---------------------------------------------------- #
        bs = blackswan_scan(all_items, frame=None)
        if not bs.get("flags"):
            fs.append(_neutral("blackswan", "黑天鹅扫描", bs.get("detail", "未发现黑天鹅信号"),
                               0.5, theory="尾部风险", horizon=("d1", "d5", "d10", "d20")))
        else:
            sev = _num(bs.get("severity"), 0.5)
            fs.append(Factor("blackswan", "黑天鹅扫描", Category.NEWS, DIR_BEAR,
                             clamp(sev, 0.3, 1.0), _reliability(all_items, 0.8),
                             value="；".join(f["type"] for f in bs["flags"]),
                             detail=bs.get("detail", ""), theory="尾部风险 / 事件驱动",
                             horizon=("d1", "d5", "d10", "d20")))
    except Exception as exc:  # noqa: BLE001
        _note("event_factors 内部异常（已降级）：%r" % exc)
        fs.append(_neutral("news_sentiment_agg", "新闻情感", "消息面因子生成异常，已降级", 0.1))
    return fs


# --------------------------------------------------------------------------- #
# 自测
# --------------------------------------------------------------------------- #
def _fake_items() -> list:
    """合成新闻/公告（覆盖各类标签与情感用例），日期相对今天。"""
    today = _today()

    def d(days):
        return (today - dt.timedelta(days=days)).strftime("%Y-%m-%d")

    return [
        {"title": "贵州茅台：2026年半年度业绩预增公告，净利润同比增长35%", "date": d(1),
         "source": "公告"},
        {"title": "控股股东拟减持不超过3%股份", "date": d(2), "source": "公告"},
        {"title": "公司关于回购股份方案的公告", "date": d(4), "source": "公告"},
        {"title": "关于筹划重大资产重组暨关联交易的提示性公告", "date": d(6), "source": "公告"},
        {"title": "公司收到中国证监会立案告知书，因涉嫌信息披露违法违规", "date": d(3),
         "source": "公告"},
        {"title": "关于限售股上市流通的提示性公告（解禁）", "date": d(8), "source": "公告"},
        {"title": "国务院常务会议部署以旧换新与专项债支持消费，白酒行业景气回升",
         "date": d(5), "source": "证券时报"},
        {"title": "某国宣布对相关产品加征关税并实施出口管制", "date": d(7), "source": "财联社"},
        {"title": "公司获得政府补贴1.2亿元，新产品中标大额订单", "date": d(9), "source": "公告"},
        {"title": "业绩不及预期，第三季度净利润大幅下滑并下调全年指引", "date": d(12),
         "source": "东方财富"},
        {"title": "关于股东部分股份质押的公告", "date": d(15), "source": "公告"},
        {"title": "公司涉及重大诉讼，涉案金额3.5亿元", "date": d(20), "source": "公告"},
    ]


def _self_test() -> None:
    """离线自测：合成新闻 + 纯解析，**不发网络请求**。"""
    print("=== newsflow self-test ===")
    items = _fake_items()

    # ---- 情感打分（复合词/否定词/程度副词） ------------------------------ #
    cases = [
        ("公司业绩大幅预增，净利润同比增长80%", 1),
        ("业绩不及预期，净利润大幅下滑并下调全年指引", -1),
        ("控股股东拟减持不超过3%股份", -1),
        ("公司关于回购股份方案的公告", 1),
        ("因涉嫌信息披露违法违规被立案调查", -1),
        ("未出现亏损", 1),                      # 否定词翻转
        ("由盈转亏，亏损扩大", -1),
        ("增收不增利", -1),
        ("中标12亿元大额订单", 1),
        ("", 0),
    ]
    print("情感打分用例：")
    for text, expect in cases:
        s = sentiment_score_text(text)
        ok = (s > 0.02 and expect > 0) or (s < -0.02 and expect < 0) or \
             (abs(s) <= 0.02 and expect == 0)
        print("   %-42s → %+.3f %s" % (text[:40] or "(空)", s, "OK" if ok else "!! 方向异常"))
        assert ok, "情感方向判定错误：%s" % text

    # ---- 分类 ------------------------------------------------------------ #
    cls = classify_news(items)
    print("分类结果（%d 条）：" % len(cls))
    for c in cls[:12]:
        print("   [%-4s] %-4s %.2f | %s" % (c["tag"],
                                            {1: "利好", -1: "利空", 0: "中性"}[c["direction"]],
                                            c["strength"], c["title"][:40]))
    tags = {c["tag"] for c in cls}
    assert {"业绩", "减持", "回购", "重组", "监管", "解禁", "政策", "宏观"} & tags

    # ---- 黑天鹅 / 政策扫描 ----------------------------------------------- #
    bs = blackswan_scan(items)
    print("黑天鹅：severity=%.2f flags=%s" % (bs["severity"],
                                              [(f["type"], f["severity"]) for f in bs["flags"]]))
    assert bs["flags"] and bs["severity"] > 0
    pol = policy_news_scan(items)
    print("政策扫描：score=%+.2f themes=%s 命中=%d 条"
          % (pol["policy_score"], pol["themes"], len(pol["items"])))

    # ---- 因子生成 -------------------------------------------------------- #
    fs = event_factors("600519", news=items, announcements=items[:4], name="贵州茅台")
    ids = sorted({f.id for f in fs})
    print("因子数量：%d" % len(fs))
    print("因子 id 清单（%d）：%s" % (len(ids), ids))
    for f in fs:
        print("   %-22s %-4s 强度 %.2f 可信度 %.2f 周期=%s | %s"
              % (f.id, {1: "看涨", -1: "看跌", 0: "中性"}[f.direction], f.strength,
                 f.reliability, "/".join(f.horizon) if f.horizon else "全部",
                 (f.detail or "")[:50]))
    assert "news_sentiment_agg" in ids and "blackswan" in ids and "unlock_pressure" in ids
    assert all(f.category == Category.NEWS for f in fs)

    # ---- 全空用例 -------------------------------------------------------- #
    empty = event_factors("600519")
    print("空数据因子数量：%d，id：%s" % (len(empty), sorted({f.id for f in empty})))
    assert all(f.reliability <= 0.5 for f in empty)
    assert classify_news([]) == [] and classify_news(None) == []
    assert blackswan_scan([])["severity"] == 0.0
    assert blackswan_scan(None)["flags"] == []
    assert policy_news_scan([])["policy_score"] == 0.0
    assert sentiment_score_text("") == 0.0 and sentiment_score_text(None) == 0.0
    print("空数据黑天鹅：", blackswan_scan([])["detail"])

    print("newsflow self-test OK")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
