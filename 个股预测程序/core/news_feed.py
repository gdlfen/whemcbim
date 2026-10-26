# -*- coding: utf-8 -*-
"""
真实新闻抓取 —— 给 AI 提供「最新的、可核查的」事实，而不是让它凭记忆编。

为什么必须这么做
----------------
把"找最新宏观事件"交给大模型的效果很差：它的训练语料有截止时间，
实际会返回 2024 年甚至更早的事件，看起来像模像样但毫无用处。

正确的架构是：
    **程序去抓真实的最新资讯 → AI 只负责筛选与解读**
这样即使模型知识陈旧，它处理的也是今天的新数据。

数据源（均已实测可用，无需 Key）
--------------------------------
· 东方财富快讯   ``newsapi.eastmoney.com/kuaixun/v1/getlist_102_...``
  返回 ``var ajaxResult={"LivesList":[{title, showtime, url_unique, ...}]}``
· 新浪财经滚动   ``feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2516...``
  返回 ``{"result":{"data":[{title, ctime(时间戳), url, ...}]}}``

失败时如实返回空列表，由上层决定"没有实时资讯"时如何提示用户，
绝不退回"用模型记忆代替"。
"""
from __future__ import annotations

import datetime as dt
import re
import time

from .utils import get_logger

log = get_logger()

H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122"}

#: 进程内缓存：避免每次点击都重复抓
_CACHE: dict = {}
_TTL = 300.0          # 5 分钟


def _cache_get(key: str):
    got = _CACHE.get(key)
    if got and (time.time() - got[0]) < _TTL:
        return got[1]
    return None


def _cache_put(key: str, val):
    _CACHE[key] = (time.time(), val)
    return val


def _clean_brief(s: str, limit: int = 180) -> str:
    """清理摘要：去掉【…】标题前缀、多余的空白与换行，并截断。"""
    t = str(s or "").strip()
    t = re.sub(r"^【[^】]{0,40}】", "", t)      # 去【标题】前缀
    t = re.sub(r"来源：[^ ]{1,20}\s*", "", t)   # 去“来源：xxx”
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit]


def _norm_time(s: str) -> str:
    """把各种时间表示统一成 ``YYYY-MM-DD HH:MM``。"""
    s = str(s or "").strip()
    if not s:
        return ""
    # 纯时间戳
    if re.fullmatch(r"\d{10,13}", s):
        ts = int(s[:10])
        try:
            return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:  # noqa: BLE001
            return ""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})[ T]?(\d{2}):(\d{2})", s)
    if m:
        return "%s-%s-%s %s:%s" % m.groups()
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return "%s-%s-%s" % m.groups()
    return s[:16]


def fetch_eastmoney_kuaixun(limit: int = 40) -> list:
    """东方财富快讯（实时、频率高）。"""
    got = _cache_get("em")
    if got is not None:
        return got[:limit]
    out = []
    try:
        import json

        import requests

        url = ("https://newsapi.eastmoney.com/kuaixun/v1/"
               "getlist_102_ajaxResult_50_1_.html")
        t = requests.get(url, timeout=12,
                         headers={**H, "Referer": "https://kuaixun.eastmoney.com/"}
                         ).text.strip()
        b = t[t.find("{"):t.rfind("}") + 1]
        j = json.loads(b)
        for it in (j.get("LivesList") or []):
            title = str(it.get("title") or it.get("simtitle") or "").strip()
            if not title:
                continue
            out.append({
                "title": title[:120],
                "time": _norm_time(it.get("showtime") or ""),
                "source": "东方财富快讯",
                "url": it.get("url_unique") or it.get("url_w") or "",
                # 快讯自带摘要（digest），有时带【标题】前缀，去掉更干净
                "brief": _clean_brief(it.get("digest") or ""),
            })
    except Exception as exc:  # noqa: BLE001
        log.debug("东财快讯抓取失败: %s", exc)
    _cache_put("em", out)
    return out[:limit]


def fetch_sina_roll(limit: int = 30) -> list:
    """新浪财经滚动新闻（偏宏观/市场）。"""
    got = _cache_get("sina")
    if got is not None:
        return got[:limit]
    out = []
    try:
        import requests

        url = ("https://feed.mix.sina.com.cn/api/roll/get?"
               "pageid=153&lid=2516&num=30&page=1")
        j = requests.get(url, timeout=12,
                         headers={**H, "Referer": "https://finance.sina.com.cn/"}
                         ).json()
        for it in ((j.get("result") or {}).get("data") or []):
            title = str(it.get("title") or "").strip()
            if not title:
                continue
            out.append({
                "title": title[:120],
                "time": _norm_time(it.get("ctime") or it.get("mtime") or ""),
                "source": "新浪财经",
                "url": it.get("url") or "",
                "brief": _clean_brief(it.get("intro") or it.get("summary") or ""),
            })
    except Exception as exc:  # noqa: BLE001
        log.debug("新浪滚动抓取失败: %s", exc)
    _cache_put("sina", out)
    return out[:limit]


# --------------------------------------------------------------------------- #
# 权威站点 HTML 列表（证监会/央行/统计局/发改委/财政部/国务院）
# --------------------------------------------------------------------------- #
#: 每个源：(标识, 中文名, 列表页 URL, 站点根，用于拼绝对链接)
GOV_SOURCES = [
    ("csrc", "证监会", "http://www.csrc.gov.cn/csrc/c100028/common_list.shtml",
     "http://www.csrc.gov.cn"),
    ("pbc", "中国人民银行",
     "http://www.pbc.gov.cn/goutongjiaoliu/113456/113469/index.html",
     "http://www.pbc.gov.cn"),
    ("stats", "国家统计局", "https://www.stats.gov.cn/sj/zxfb/",
     "https://www.stats.gov.cn"),
    ("ndrc", "国家发改委", "https://www.ndrc.gov.cn/xwdt/xwfb/",
     "https://www.ndrc.gov.cn"),
    ("mof", "财政部",
     "https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/",
     "https://www.mof.gov.cn"),

]

#: 非新闻标题黑名单（导航/页脚/备案等）
BAD_TITLES = ("English", "ICP备", "公网安备", "网站地图", "联系我们",
              "版权声明", "无障碍", "登录", "注册", "首页", "更多",
              "网站声明", "关于我们", "简体", "繁体")
#: 非新闻链接黑名单
BAD_URLS = ("beian.miit.gov.cn", "beian.gov.cn", "/en/", "javascript:")

#: 日期正则：2026-10-26 / 2026/10/26 / 2026年10月26日 / [2026-10-26]
_DATE_RE = re.compile(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})")
#: 锚点正则（同时抓纯文本，用于取标题）
_A_RE = re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                   re.S | re.I)


def _strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    s = (s.replace("&nbsp;", " ").replace("&amp;", "&")
          .replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"'))
    return re.sub(r"\s+", " ", s).strip()


def _abs_url(href: str, root: str, page: str) -> str:
    href = (href or "").strip()
    if not href or href.startswith(("javascript:", "#", "mailto:")):
        return ""
    if href.startswith("http"):
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return root.rstrip("/") + href
    # 相对路径：以页面目录为基准
    base = page[:page.rfind("/") + 1]
    return base + href


def _parse_gov_list(html: str, page: str, root: str, source: str,
                    limit: int = 30) -> list:
    """从政府站列表页解析「标题 + 日期 + 原文链接」。

    通用做法：遍历所有锚点，取纯文本作标题；日期在锚点自身或其后 200 字符内
    查找（多数政府站把日期放在标题后的 span 里）。
    """
    out, seen = [], set()
    for m in _A_RE.finditer(html):
        href, inner = m.group(1), m.group(2)
        title = _strip_tags(inner)
        if len(title) < 8 or len(title) > 90:
            continue
        # 日期：锚点内 → 其后 240 字符
        seg = inner + html[m.end():m.end() + 240]
        if not _DATE_RE.search(seg):
            seg = html[max(0, m.start() - 240):m.start()] + seg
        dm = _DATE_RE.search(seg)
        date = ("%s-%02d-%02d" % (dm.group(1), int(dm.group(2)),
                                  int(dm.group(3)))) if dm else ""
        url = _abs_url(href, root, page).replace("/./", "/")
        if not url or not date:
            continue
        # 过滤导航/页脚/备案等非新闻链接
        if any(bad in title for bad in BAD_TITLES) or any(
                bad in url for bad in BAD_URLS):
            continue
        k = title[:30]
        if k in seen:
            continue
        seen.add(k)
        out.append({"title": title[:120], "time": date, "source": source,
                    "url": url, "brief": ""})
        if len(out) >= limit:
            break
    return out


def fetch_gov_source(key: str, limit: int = 25) -> list:
    """抓取单个权威站点的列表页。失败返回空列表。"""
    got = _cache_get("gov_" + key)
    if got is not None:
        return got[:limit]
    cfg = next((x for x in GOV_SOURCES if x[0] == key), None)
    if cfg is None:
        return []
    _, name, page, root = cfg
    out = []
    try:
        import requests

        r = requests.get(page, timeout=15,
                         headers={**H, "Referer": root + "/"})
        # 政府站常见 GB2312/GBK，用 apparent_encoding 纠正
        if not r.encoding or r.encoding.lower() in ("iso-8859-1", "ascii"):
            r.encoding = r.apparent_encoding or "utf-8"
        out = _parse_gov_list(r.text, page, root, name, limit=limit)
    except Exception as exc:  # noqa: BLE001
        log.debug("抓取 %s 失败: %s", name, exc)
    _cache_put("gov_" + key, out)
    return out[:limit]


def fetch_gov_all(limit_each: int = 20) -> dict:
    """抓取全部权威站点，按日期倒序合并。"""
    items, sources = [], []
    for key, name, _page, _root in GOV_SOURCES:
        try:
            got = fetch_gov_source(key, limit_each)
        except Exception:  # noqa: BLE001
            got = []
        if got:
            items.extend(got)
            sources.append(name)
    items = [x for x in items if x.get("time")]
    items.sort(key=lambda x: x.get("time") or "", reverse=True)
    return {"items": items, "sources": sources,
            "detail": ("已抓取 %d 条权威发布（%s）"
                       % (len(items), "、".join(sources)) if sources else
                       "未能获取权威发布（网络或接口不可用）")}


def fetch_headlines(limit: int = 50) -> dict:
    """抓取全部可用源，按时间倒序合并。

    Returns
    -------
    dict(items, sources, detail)
    """
    items, sources = [], []
    # 权威站点优先（含原文链接，用于核实政策与统计数据）
    try:
        gov = fetch_gov_all(18)
        if gov.get("items"):
            items.extend(gov["items"])
            sources.extend(gov.get("sources") or [])
    except Exception as exc:  # noqa: BLE001
        log.debug("抓取权威站点失败: %s", exc)
    for fn, name in ((fetch_eastmoney_kuaixun, "东方财富快讯"),
                     (fetch_sina_roll, "新浪财经")):
        try:
            got = fn(limit)
        except Exception:  # noqa: BLE001
            got = []
        if got:
            items.extend(got)
            sources.append(name)
    # 权威站保留固定名额：它们的日期只到天（如 "2026-10-26"），
    # 若与快讯（精确到分钟）一起倒序，会被秒级快讯挤出列表 —— 而这些
    # 恰恰是政策/统计的原始发布，必须保留。
    _gov = [x for x in items if x.get("source") in
            {n for _k, n, _p, _r in GOV_SOURCES}]
    _news = [x for x in items if x not in _gov]
    _gov.sort(key=lambda x: x.get("time") or "", reverse=True)
    _news.sort(key=lambda x: x.get("time") or "", reverse=True)
    _keep_gov = min(len(_gov), 15)
    items = _gov[:_keep_gov] + _news[:max(1, limit - _keep_gov)]
    # 去重（同标题）
    seen, uniq = set(), []
    for it in items:
        k = it["title"][:40]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(it)
    return {"items": uniq[:limit], "sources": sources,
            "detail": ("已抓取 %d 条实时资讯（来源：%s）"
                       % (len(uniq[:limit]), "、".join(sources))
                       if sources else "未能获取实时资讯（网络或接口不可用）")}


if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")

    print("=== 1. 东财快讯 ===")
    em = fetch_eastmoney_kuaixun(5)
    print("  %d 条" % len(em))
    for it in em[:3]:
        print("   ", it["time"], "|", it["title"][:40])

    print("\n=== 2. 新浪滚动 ===")
    sn = fetch_sina_roll(5)
    print("  %d 条" % len(sn))
    for it in sn[:3]:
        print("   ", it["time"], "|", it["title"][:40])

    print("\n=== 3. 合并 ===")
    h = fetch_headlines(20)
    print(" ", h["detail"])
    assert h["items"], "应至少抓到 1 条（若失败请检查网络）"
    times = [x["time"] for x in h["items"] if x["time"]]
    print("  最新时间:", times[0] if times else "—")
    # 时间越新越好：应包含今年的日期
    year = str(dt.date.today().year)
    recent = [x for x in h["items"] if (x.get("time") or "").startswith(year)]
    print("  本年度条数: %d/%d" % (len(recent), len(h["items"])))
    assert recent, "应抓到本年度资讯（否则数据源失效）"

    print("\n=== 4. 缓存 ===")
    t0 = time.time()
    fetch_headlines(5)
    print("  二次抓取耗时 %.3f 秒（应接近 0）" % (time.time() - t0))

    print("\n=== 5. 时间归一 ===")
    assert _norm_time("2026-10-26 18:45:56") == "2026-10-26 18:45"
    assert _norm_time("1793011012")[:4] in ("2026", "2027")
    assert _norm_time("2026-10-26") == "2026-10-26"
    print("  时间解析正确")

    print("\nnews_feed self-test OK")
