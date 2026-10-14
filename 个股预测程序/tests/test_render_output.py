# -*- coding: utf-8 -*-
"""
端到端渲染校验 —— 真正"跑一遍界面"，然后把**渲染出来的文本**逐条扫描。

为什么必须做这个
----------------
前三轮用户反馈的问题（英文残留 / None 字样 / 原始字典 / 长串小数）都属于
"代码没错、但渲染出来的字不对"。静态检查（字段标签、渲染规则）只能覆盖一部分，
真正可靠的验证是：**把界面渲染一遍，检查输出文本**。

做法
----
* 用 streamlit stub 拦截 st.markdown / write / caption / json / dataframe / metric，
  把所有"会显示给用户"的文本收集起来；
* 用真实分析结果（缓存到 tests/.payload_cache.json，避免每次联网）跑完 9 个页签；
* 对收集到的文本做"毒性扫描"：
    原始字典（"{'" / "'}"）、None、nan、inf、裸键名（xxx_yyy 形式）、
    疑似长串小数（小数点后 >8 位）。

用法：``python tests/test_render_output.py [--refresh]``
    --refresh 强制重新联网分析并刷新缓存
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, "tests", ".payload_cache.json")

# --------------------------------------------------------------------------- #
# 1) 带"采集"能力的 streamlit stub
# --------------------------------------------------------------------------- #
sys.path.insert(0, os.path.join(HERE, "tests"))
from test_app_logic import _mk_stub  # noqa: E402

CAPTURED: list = []
_stub = _mk_stub()


def _rec(name):
    def fn(*a, **k):
        # 位置参数与关键字参数都要收（st.markdown(body, unsafe_allow_html=True)
        # 正文在位置参数；st.write(body=...) 之类在关键字里）
        for x in list(a) + list(k.values()):
            if isinstance(x, str):
                CAPTURED.append(x)
            elif x is not None and not hasattr(x, "columns"):
                try:
                    CAPTURED.append(str(x))
                except Exception:  # noqa: BLE001
                    pass
    fn.__name__ = name
    return fn


for _m in ("markdown", "caption", "write", "info", "success", "warning", "error",
           "title", "header", "subheader", "code", "json", "table", "dataframe",
           "metric", "text"):
    setattr(_stub, _m, _rec(_m))
# 注意：test_app_logic 在导入时已把它那份 stub 塞进 sys.modules，
# 所以必须在 import test_app_logic **之后**再覆盖一次。
sys.modules["streamlit"] = _stub

import app  # noqa: E402
import pandas as pd  # noqa: E402

# 再确认一次 app 用的是带采集能力的 stub（否则采集恒为 0 段，测试形同虚设）
app.st = _stub
assert app.st is _stub, "app.st 未指向采集用 stub，渲染结果不会被采集"


# --------------------------------------------------------------------------- #
# 2) 取一份真实分析结果（缓存，避免每次联网）
# --------------------------------------------------------------------------- #
def get_payload(refresh: bool = False) -> dict:
    if not refresh and os.path.exists(CACHE):
        try:
            with io.open(CACHE, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            pass
    print("（首次运行：联网分析 603858 并缓存结果，约需 1~2 分钟）")
    p = app.cached_analysis("603858", False, False, 500)
    p.pop("charts", None)          # 图片 base64 太大，渲染校验不需要
    try:
        with io.open(CACHE, "w", encoding="utf-8") as fh:
            json.dump(p, fh, ensure_ascii=False, default=str)
    except Exception as exc:  # noqa: BLE001
        print("（缓存写入失败，不影响本次校验：%s）" % exc)
    return p


# --------------------------------------------------------------------------- #
# 3) 毒性扫描
# --------------------------------------------------------------------------- #
RAW_DICT = re.compile(r"\{\s*'[^']+'\s*:")
BARE_KEY = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+){1,}$")       # 形如 abc_def
# 只查这几个绝不该出现的标记。
# 注意：不要把 True/False 放进来——布尔值经 _short() 会变成是/否，
# 而 BAD_WORDS 用的是子串匹配，会把英文单词里的片段也误判（曾误报 91 处）。
BAD_WORDS = ("None", "nan", "NaN", "inf", "-inf", "null")
LONG_DEC = re.compile(r"\d+\.\d{9,}")
#: 允许出现的英文缩写（金融/技术指标/宏观），不视为"英文残留"
ALLOW_EN = {
    "pe", "pb", "ps", "pcf", "peg", "roe", "roa", "eps", "bps", "ttm", "yoy",
    "ma5", "ma10", "ma20", "ma30", "ma60", "ma120", "ma250",
    "macd", "kdj", "rsi", "boll", "sar", "dmi", "obv", "bias", "cci", "wr",
    "trix", "dma", "roc", "emv", "asi", "vr", "ar", "br", "cr", "psy", "mtm",
    "atr", "expma", "bbi", "dkx", "ene", "cpi", "ppi", "pmi", "gdp", "m2",
    "lpr", "shibor", "vix", "a50", "etf", "st", "ipo", "qfii", "rqfii",
    "beta", "roe%", "k", "d", "j", "pb", "ps", "pcf", "ev/ebitda", "ebitda",
    "w2/w1", "w3/w1", "w1", "w2", "w3", "abc", "zigzag", "atr%", "atr14",
    "pe(ttm)", "pe(静态)", "pb(mrq)", "ps(ttm)", "pcf(ttm)", "ev/ebitda",
}


def is_english_label(lab: str) -> bool:
    s = lab.strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_ ()%/.·-]*", s):
        return False
    return s.lower().strip() not in ALLOW_EN


def scan_texts() -> list:
    problems = []
    for t in CAPTURED:
        s = str(t)
        if not s.strip():
            continue
        if RAW_DICT.search(s):
            problems.append(("原始字典字样", s[:110]))
            continue
        for w in BAD_WORDS:
            if re.search(r"(?<![A-Za-z_])%s(?![A-Za-z_])" % re.escape(w), s):
                problems.append(("%s 字样" % w, s[:110]))
                break
        if LONG_DEC.search(s):
            problems.append(("长串小数", s[:110]))
    return problems


# --------------------------------------------------------------------------- #
# 4) 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    refresh = "--refresh" in sys.argv
    payload = get_payload(refresh)
    CAPTURED.clear()
    app.render_dimensions(payload)
    texts = [t for t in CAPTURED if isinstance(t, str) and t.strip()]
    print("已渲染 9 个页签，采集到 %d 段界面文本" % len(texts))

    # ① 毒性扫描
    problems = scan_texts()
    # ② 字段标签扫描：卡片里的 .lbl 必须是中文（或白名单缩写）
    labels = re.findall(r"<div class='lbl'>([^<]*)</div>", "\n".join(texts))
    bad_labels = sorted({x for x in labels if is_english_label(x)})
    # ③ 原始键名扫描：形如 abc_def 的裸键
    bare = sorted({x.strip() for x in labels if BARE_KEY.match(x.strip())})

    print("检查标签 %d 个（去重 %d）" % (len(labels), len(set(labels))))
    ok = True
    if problems:
        ok = False
        print("\n[失败] 界面文本存在 %d 处问题：" % len(problems))
        seen = set()
        for kind, s in problems:
            k = (kind, s[:60])
            if k in seen:
                continue
            seen.add(k)
            print("   [%s] %s" % (kind, s.replace("\n", " ")[:100]))
    if bad_labels:
        ok = False
        print("\n[失败] 以下标签仍是英文（未加白名单则视为残留）：%s" % bad_labels)
    if bare:
        ok = False
        print("\n[失败] 以下标签是裸键名（英文内部键泄漏）：%s" % bare)
    if ok:
        print("\n[通过] 9 个页签渲染文本无 英文残留 / None / nan / 原始字典 / 长串小数")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
