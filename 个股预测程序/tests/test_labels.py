# -*- coding: utf-8 -*-
"""
字段标签校验：界面上不应出现"本可以翻译却仍是英文"的字段名。

背景（真实用户反馈）：分析版面里散落着 `up_count`、`index_pe_percentile`、
`swing_points` 这类内部英文键名，用户看到的是英文表格。

**允许保留的英文**（中国文化里约定俗成、写成中文反而不专业的缩写）：
    MA5/MA10/MA20/MA30/MA60/MA120/MA250   均线
    PE / PB / PS / PCF / EV-EBITDA / ROE / ROA / EPS / BPS   估值与财务比率
    CPI / PPI / PMI / GDP / M2 / LPR / Shibor / VIX / A50     宏观与指数缩写
    RSI / MACD / KDJ / BOLL / SAR / CCI / WR / DMI / OBV / VR / ASI / EMV
    TRIX / DMA / ROC / BIAS / BBI / EXPMA / DKX / ENE / CR / PSY / MTM / ATR
    ETF / A股 / H股 / ST / IPO / QFII / RQFII / TTM / YoY / MoM
    R² / β / σ

用法：``python tests/test_labels.py``
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

#: 允许保留英文的缩写（大小写不敏感、按整串匹配）
ALLOWED = {
    "ma5", "ma10", "ma20", "ma30", "ma60", "ma120", "ma250",
    "pe", "pb", "ps", "pcf", "peg", "roe", "roa", "eps", "bps",
    "ev/ebitda", "ev_ebitda", "ebitda", "ttm", "yoy", "mom",
    "cpi", "ppi", "pmi", "gdp", "m2", "lpr", "shibor", "vix", "a50",
    "rsi", "macd", "kdj", "boll", "sar", "cci", "wr", "dmi", "obv", "vr",
    "asi", "emv", "trix", "dma", "roc", "bias", "bbi", "expma", "dkx", "ene",
    "cr", "psy", "mtm", "atr", "pdi", "mdi", "adx", "adxr", "asi",
    "etf", "st", "ipo", "qfii", "rqfii", "roe%", "pe(ttm)", "pe(静态)",
    "pb(mrq)", "beta", "r²", "σ", "β", "atr14", "ma", "vpt",
}

#: 明确不需要检验的内部键（渲染层已过滤，不会出现在界面上）
INTERNAL_OK = {"source", "as_of", "symbol", "code", "name", "date"}


def looks_english(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ ()%/.·-]*", str(s).strip()))


def main() -> int:
    # 用 streamlit stub 载入 app，避免依赖真实 streamlit
    import warnings

    warnings.filterwarnings("ignore")
    sys.argv = ["t"]
    stub = os.path.join(HERE, "tests", "test_app_logic.py")
    src = open(stub, encoding="utf-8").read().split("# 1)")[0]
    g = {"__file__": stub, "__name__": "stub"}
    exec(compile(src, "stub", "exec"), g)          # noqa: S102
    import app                                     # noqa: E402

    labels = app._FIELD_LABELS
    bad = []
    for k, v in labels.items():
        if v is None:
            continue
        if not looks_english(v):
            continue
        if str(v).strip().lower() in ALLOWED or str(k).lower() in ALLOWED:
            continue
        bad.append((k, v))

    print("字段标签总数：%d（其中 %d 个标记为不展示）"
          % (len(labels), sum(1 for v in labels.values() if v is None)))
    print("允许保留的英文缩写：%d 个" % len(ALLOWED))
    if bad:
        print("\n[失败] 以下字段的中文标签其实还是英文，应补中文"
              "（若确属约定俗成的缩写，请加入 ALLOWED 白名单）：")
        for k, v in sorted(bad):
            print("   %-24s -> %r" % (k, v))
        return 1
    print("\n[通过] 界面字段标签无遗留英文（金融缩写已白名单放行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
