# -*- coding: utf-8 -*-
"""
上下游商品价格 —— **按个股所属行业筛选**，只展示与该公司真正相关的品种。

问题背景
--------
早期版本把所有商品（原油/铜/黄金/螺纹钢/碳酸锂/铁矿石）一股脑塞进"上下游商品价格"，
于是分析一家中药企业时也会看到"原油、铜、黄金"——这些和它的成本、售价毫无关系，
属于典型的"采集了无关数据"。

解决思路
--------
1. 维护一份**行业关键词 → 相关商品**的映射（煤化工/有色/农牧/医药/新能源…）；
2. 用个股的**行业与概念板块名称**去匹配关键词，命中哪些行业就取哪些商品；
3. 匹配不到时**不臆造**：标记为"该行业未匹配到直接相关商品"，
   只保留可作为**宏观成本环境参考**的宽泛指标（并把这一点写清楚）。

商品代码全部经过实测（akshare 国内期货主力连续合约）。
"""

from __future__ import annotations

import math

from . import config
from .contracts import now_cn, safe_float
from .utils import cache_key, cached, get_logger

log = get_logger()

#: 商品名 → akshare 主力连续合约代码（均已实测可用）
COMMODITY_FUTURES: dict[str, str] = {
    # 能源
    "动力煤": "ZC0", "焦煤": "JM0", "焦炭": "J0", "原油(内盘)": "SC0", "液化石油气": "PG0",
    # 化工
    "PTA": "TA0", "甲醇": "MA0", "聚丙烯": "PP0", "PVC": "V0", "乙二醇": "EG0",
    "纯碱": "SA0", "玻璃": "FG0", "尿素": "UR0", "短纤": "PF0", "橡胶": "RU0",
    # 黑色
    "螺纹钢": "RB0", "铁矿石": "I0", "热轧卷板": "HC0", "不锈钢": "SS0",
    # 有色
    "沪铜": "CU0", "沪铝": "AL0", "沪锌": "ZN0", "沪镍": "NI0", "沪铅": "PB0",
    "沪锡": "SN0", "沪金": "AU0", "沪银": "AG0", "工业硅": "SI0", "碳酸锂": "LC0",
    # 农牧
    "豆粕": "M0", "豆油": "Y0", "棕榈油": "P0", "玉米": "C0", "玉米淀粉": "CS0",
    "白糖": "SR0", "棉花": "CF0", "苹果": "AP0", "生猪": "LH0", "鸡蛋": "JD0",
    "菜籽油": "OI0", "菜籽粕": "RM0", "花生": "PK0", "红枣": "CJ0",
}

#: 行业/概念关键词 → 相关商品（按"越具体越靠前"排列；同一商品可被多个行业引用）
INDUSTRY_COMMODITY_MAP: dict[tuple, tuple] = {
    # —— 医药 / 中药 ——
    ("中药", "中医药", "医药生物", "生物制药", "化学制药", "原料药", "创新药",
     "医疗器械", "医疗服务", "疫苗", "生物疫苗"): ("玉米", "玉米淀粉", "玻璃", "纯碱"),
    # —— 白酒 / 食品饮料 / 调味品 ——
    ("白酒", "食品饮料", "调味品", "调味发酵", "啤酒", "乳业", "食品加工",
     "软饮料", "烘焙", "预制菜"): ("玉米", "白糖", "豆油", "棕榈油", "玻璃", "纯碱"),
    # —— 农牧 ——
    ("农牧", "养殖", "饲料", "猪肉", "鸡肉", "农业", "种植", "水产", "禽"): (
        "豆粕", "玉米", "生猪", "鸡蛋", "菜籽粕", "豆油"),
    # —— 煤炭 ——
    ("煤炭", "焦炭", "煤化工", "动力煤"): ("动力煤", "焦煤", "焦炭"),
    # —— 石油 / 化工 ——
    ("石油", "油气", "炼化", "油服", "化工", "化学制品", "化学原料", "石化",
     "聚氨酯", "涂料", "橡胶", "塑料", "化纤"): (
        "原油(内盘)", "PTA", "甲醇", "聚丙烯", "PVC", "乙二醇", "橡胶"),
    # —— 有色 ——
    ("黄金", "贵金属", "白银"): ("沪金", "沪银"),
    ("铜", "有色", "工业金属", "铜产业"): ("沪铜", "沪铝", "沪锌"),
    ("铝", "铝产业"): ("沪铝",),
    ("锂", "新能源", "锂电池", "电池", "储能", "正极", "负极", "盐湖"): (
        "碳酸锂", "工业硅"),
    ("稀土", "小金属", "钨", "钼", "锡", "镍"): ("沪镍", "沪锡"),
    # —— 钢铁 / 建材 ——
    ("钢铁", "普钢", "特钢", "钢管"): ("螺纹钢", "铁矿石", "热轧卷板", "焦炭"),
    ("水泥", "建材", "玻璃", "玻璃纤维"): ("玻璃", "纯碱"),
    # —— 新能源 / 电力 ——
    ("光伏", "硅料", "半导体", "芯片", "电子"): ("工业硅", "沪银"),
    ("电力", "火电", "热电"): ("动力煤",),
    # —— 纺服 / 轻工 ——
    ("纺织", "服装", "家纺", "棉"): ("棉花", "PTA", "短纤"),
    # —— 汽车 / 机械 ——
    ("汽车", "汽车零部件", "机械", "工程机械", "机器人"): ("螺纹钢", "沪铝", "沪铜"),
    # —— 造纸 / 包装 ——
    ("造纸", "包装", "印刷"): ("玻璃", "PVC", "纯碱"),
    # —— 交运 / 航空 ——
    ("航空", "机场", "航运", "物流", "快递"): ("原油(内盘)",),
}

#: 匹配不到行业时，只给这些"宏观成本环境"参考项，并明确标注与个股无直接关系
FALLBACK_MACRO_COMMODITIES = ("原油(内盘)", "沪铜")


def _ak_futures(code: str) -> dict:
    """取单个期货主力连续合约的最新价（akshare）。失败返回 {}。"""
    try:
        import akshare as ak

        df = ak.futures_main_sina(symbol=code)
        if df is None or not len(df):
            return {}
        df = df.dropna()
        if not len(df):
            return {}
        last = df.iloc[-1]
        cols = list(df.columns)
        price = safe_float(last[cols[1]] if len(cols) > 1 else None)
        if price != price or price <= 0:
            return {}
        prev = safe_float(last[cols[2]]) if len(cols) > 2 else float("nan")
        high = safe_float(last[cols[3]]) if len(cols) > 3 else float("nan")
        low = safe_float(last[cols[4]]) if len(cols) > 4 else float("nan")
        pct = ((price / prev - 1) * 100) if (prev == prev and prev) else float("nan")
        return {"price": price, "prev_close": prev, "high": high, "low": low,
                "pct_chg": pct, "date": str(last[cols[0]])[:10],
                "name": code, "source": "akshare.futures_main_sina"}
    except Exception as exc:  # noqa: BLE001
        log.debug("期货 %s 取数失败: %s", code, exc)
        return {}


def fetch_extended_commodities(force: bool = False, max_items: int = 12,
                               names=None) -> dict:
    """取商品价格。

    Parameters
    ----------
    names : iterable | None
        **优先要取的品种**。为了效率，应当先根据行业算出需要的品种再取数
        （否则"先随便取 N 个再筛选"会漏掉真正相关的品种——这是早期实现的错误）。
    max_items : 除优先品种外，最多再补取几个常见品种。

    Returns
    -------
    dict：中文商品名 → 价格信息；另含 ``"detail"``。
    """
    want = [str(x) for x in (names or []) if x]
    key = cache_key("commodity_ext", ",".join(sorted(want)), now_cn().strftime("%Y%m%d%H"))

    def _p() -> dict:
        out: dict = {}
        try:
            from . import macro as M

            base = M.fetch_commodity_prices(force=force) or {}
            for k, v in base.items():
                if isinstance(v, dict) and k not in ("alias", "missing", "detail"):
                    out[k] = v
        except Exception as exc:  # noqa: BLE001
            log.debug("基础商品取数失败: %s", exc)
        # ① 先取"行业需要但基础集里没有"的品种
        todo = [(n, COMMODITY_FUTURES[n]) for n in want
                if n in COMMODITY_FUTURES and n not in out]
        # ② 再按需补取几个常见品种（够用即可，避免请求过多）
        extra = [(n, c) for n, c in COMMODITY_FUTURES.items()
                 if n not in out and n not in want]
        todo += extra[:max(0, max_items)]
        for name, code in todo:
            got = _ak_futures(code)
            if got:
                out[name] = got
        out["detail"] = "共取到 %d 个商品品种" % sum(
            1 for k, v in out.items() if isinstance(v, dict))
        return out

    try:
        if force:
            return _p()
        got = cached(key, 6 * 3600, _p)
        return got if isinstance(got, dict) else {}
    except Exception as exc:  # noqa: BLE001
        log.debug("fetch_extended_commodities 降级: %s", exc)
        return {}


def matched_industries(industry_names) -> list:
    """返回命中的行业关键词列表（按映射表顺序）。"""
    text = " ".join(str(x) for x in (industry_names or []) if x)
    if not text:
        return []
    hits = []
    for keys, _ in INDUSTRY_COMMODITY_MAP.items():
        for k in keys:
            if k in text and k not in hits:
                hits.append(k)
    return hits


def needed_commodities(industry_names) -> list:
    """先算出该行业**需要哪些商品**（取数前调用，避免取一堆无关品种）。"""
    hits = matched_industries(industry_names)
    want: list = []
    for k in hits:
        for keys, names in INDUSTRY_COMMODITY_MAP.items():
            if k in keys:
                for nm in names:
                    if nm not in want:
                        want.append(nm)
                break
    if not want:
        want = list(FALLBACK_MACRO_COMMODITIES)
    return want


def relevant_commodities(commodities: dict | None, industry_names=None,
                         max_items: int = 6, auto_fetch: bool = True) -> dict:
    """挑出与**该行业**相关的商品；``commodities`` 为空且 ``auto_fetch`` 时会自动取数。

    Parameters
    ----------
    commodities : :func:`fetch_extended_commodities` 的结果，可为 None
    industry_names : 个股的行业/概念板块名称（list 或 str）
    max_items : 最多保留几个品种
    auto_fetch : 缺数据时是否按行业需要去取（只取需要的品种，效率高）

    Returns
    -------
    dict：``{"items": {商品名: 数据}, "industries": [...], "related": bool, "detail": 说明}``
    """
    hits = matched_industries(industry_names)
    want = needed_commodities(industry_names)
    items = {k: v for k, v in (commodities or {}).items()
             if isinstance(v, dict) and k not in ("alias", "missing")}
    if auto_fetch and not all(w in items for w in want[:2]):
        try:
            more = fetch_extended_commodities(names=want, max_items=0)
            for k, v in more.items():
                if isinstance(v, dict):
                    items[k] = v
        except Exception as exc:  # noqa: BLE001
            log.debug("按需取商品失败: %s", exc)

    picked: dict = {}
    for nm in want:
        if nm in items and nm not in picked:
            picked[nm] = items[nm]
    related = bool(picked) and bool(hits)
    if not picked:
        for nm in FALLBACK_MACRO_COMMODITIES:
            if nm in items:
                picked[nm] = items[nm]
    picked = dict(list(picked.items())[:max_items])

    if not items:
        detail = "未取到任何商品价格（期货接口不可用）"
    elif related:
        detail = ("已按行业「%s」筛选出 %d 个相关品种：%s"
                  % ("、".join(hits[:4]), len(picked), "、".join(picked)))
    else:
        detail = ("未匹配到与「%s」直接相关的上游商品，"
                  "以下仅为宏观成本环境参考（与个股无直接关系）"
                  % ("、".join(str(x) for x in (industry_names or [])[:3]) or "该行业"))
    return {"items": picked, "industries": hits, "related": related, "detail": detail}


def commodity_factors(commodities: dict, industry_names=None) -> list:
    """把相关商品的价格变化转成因子（只对相关品种生效）。

    逻辑：原材料上涨 → 对**下游制造/消费**企业是成本压力（利空）；
    对**上游资源**企业是涨价利好。这里结合命中的行业关键词判断方向。
    """
    from .contracts import Category, DIR_BEAR, DIR_BULL, Factor

    out: list[Factor] = []
    rel = relevant_commodities(commodities, industry_names)
    items = rel.get("items") or {}
    if not items:
        return out
    hits = rel.get("industries") or []
    # 上游资源型行业：商品涨价 = 利好
    upstream_kw = ("煤炭", "有色", "黄金", "铜", "铝", "锂", "稀土", "小金属",
                   "钢铁", "石油", "油气", "农业", "种植", "养殖", "化工", "化学原料")
    is_upstream = any(any(u in h for u in upstream_kw) for h in hits)

    pcts = []
    for name, d in items.items():
        p = safe_float((d or {}).get("pct_chg"))
        if p == p:
            pcts.append((name, p))
    if not pcts:
        return out
    avg = sum(p for _, p in pcts) / len(pcts)
    direction = 0
    if avg > 1.0:
        direction = DIR_BULL if is_upstream else DIR_BEAR
    elif avg < -1.0:
        direction = DIR_BEAR if is_upstream else DIR_BULL
    strength = min(abs(avg) / 4.0, 0.7)
    detail = "相关商品（%s）当日平均%s %.2f%%；%s" % (
        "、".join(n for n, _ in pcts[:4]), "上涨" if avg > 0 else "下跌", abs(avg),
        "该股属上游资源型，涨价利好" if is_upstream else
        ("该股属下游制造/消费型，原材料上涨形成成本压力" if avg > 0
         else "该股属下游制造/消费型，原材料下跌降低成本"))
    out.append(Factor(
        # 用独立 id，避免被 macro 模块里"全品类商品"的同名因子覆盖
        id="commodity_cost_industry", name="上下游商品价格（行业相关）",
        category=Category.MACRO,
        direction=direction, strength=strength, reliability=0.6 if rel["related"] else 0.35,
        value="；".join("%s %+.2f%%" % (n, p) for n, p in pcts[:4]),
        detail=detail, theory="上下游供需/成本传导",
        horizon=("d20", "m1", "q1", "h1", "y1")))
    return out


if __name__ == "__main__":  # pragma: no cover
    print("=== 商品品种与行业映射自测 ===")
    print("可选商品 %d 种" % len(COMMODITY_FUTURES))
    cases = [
        (["中药Ⅲ", "中药Ⅱ", "医药生物"], "步长制药（中药）"),
        (["白酒Ⅲ", "食品饮料"], "贵州茅台（白酒）"),
        (["煤炭开采", "动力煤"], "中国神华（煤炭·上游）"),
        (["锂电池", "储能"], "宁德时代（锂电）"),
        (["光伏设备", "硅料"], "隆基绿能（光伏）"),
        (["钢铁", "普钢"], "宝钢股份（钢铁·上游）"),
        ([], "未知行业"),
    ]
    for names, label in cases:
        r = relevant_commodities({k: {"pct_chg": 1.0, "price": 1.0}
                                  for k in list(COMMODITY_FUTURES)},
                                 names)
        print("  %-22s 相关=%-5s 命中=%-14s 品种=%s"
              % (label, r["related"], "/".join(r["industries"][:3]) or "—",
                 "、".join(r["items"]) or "—"))
    # 断言：中药企业不应匹配到原油/铜
    r = relevant_commodities({k: {"pct_chg": 0.5, "price": 1.0}
                              for k in COMMODITY_FUTURES}, ["中药Ⅲ", "医药生物"])
    assert r["related"], "中药行业应有匹配商品"
    assert "原油(内盘)" not in r["items"], "中药企业不应出现原油"
    assert "沪铜" not in r["items"], "中药企业不应出现铜"
    print("\n断言通过：中药企业不会匹配到原油/铜等无关商品")

    print("\n=== 联网实测（按行业需要取数） ===")
    for names in (["中药Ⅲ", "医药生物"], ["煤炭开采"], ["白酒Ⅲ"], []):
        rr = relevant_commodities(None, names)
        print("  %-18s -> %-28s | %s" % ("/".join(names) or "（空）",
                                        "、".join(rr["items"]) or "—",
                                        rr["detail"][:64]))
    print("\ncommodity_industry self-test OK")
