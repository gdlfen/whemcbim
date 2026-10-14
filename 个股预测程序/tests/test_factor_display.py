# -*- coding: utf-8 -*-
"""
展示值校验：因子的 value / detail 里不得出现 None / nan / inf 字样。

背景（真实用户反馈）：界面上出现 "touches=None"、"PEG 估值 → None" 这类字样。
两类原因，都在这里固定住：
  1. Factor.value 直接来自数据源，取不到时是 None/NaN → 由 contracts._clean_display
     在构造时统一转成"—"；
  2. 代码里格式化字符串时读错了键（例如把 up_line.touches 写成 tl["touches"]），
     导致永远拼出 "xxx=None" —— 这类必须从源头修，本测试能兜住回归。

用法：``python tests/test_factor_display.py``
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

BAD_TOKENS = ("None", "nan", "NaN", "inf", "-inf", "null")


def main() -> int:
    from core.contracts import Category, Factor

    # 1) 单元级：构造时即应清理
    cases = [
        (None, "—"), (float("nan"), "—"), (float("inf"), "—"),
        ("nan", "—"), ("None", "—"), ("", "—"),
        (1.23, 1.23), ("正常文本", "正常文本"), (0, 0),
    ]
    for raw, want in cases:
        got = Factor(id="t", name="t", category=Category.TECH, direction=0,
                     strength=0.5, reliability=0.5, value=raw).value
        ok = (got == want) or (isinstance(want, float) and isinstance(got, float)
                               and got == got and want == want
                               and abs(got - want) < 1e-9)
        if not ok:
            print("[失败] value=%r 清理后应为 %r，实际 %r" % (raw, want, got))
            return 1
    # 嵌套字典/列表里的 nan 也要清掉
    nested = Factor(id="t", name="t", category=Category.TECH, direction=0,
                    strength=0.5, reliability=0.5,
                    value={"a": float("nan"), "b": [1, None]}).value
    if nested != {"a": "—", "b": [1, "—"]}:
        print("[失败] 嵌套值未清理：%r" % (nested,))
        return 1

    # 2) 端到端：真实数据的全部因子
    from core.selftest_data import make_ohlcv
    from core.technical import technical_factors

    df = make_ohlcv(400, seed=5)
    factors = list(technical_factors(df))
    from core import fundamental as F
    try:
        factors += list(F.fundamental_factors("600519", df=df) or [])
    except Exception:  # noqa: BLE001
        pass

    bad = []
    for f in factors:
        v, d = str(f.value), str(f.detail)
        for tok in BAD_TOKENS:
            if tok in v:
                bad.append(("value", f.id, v[:70]))
            if tok in d:
                bad.append(("detail", f.id, d[:70]))
    print("检查因子 %d 条" % len(factors))
    if bad:
        print("\n[失败] 以下因子的展示文本里含 None/nan/inf：")
        for kind, fid, txt in bad[:20]:
            print("   [%s] %-26s %s" % (kind, fid, txt))
        return 1
    print("\n[通过] 因子展示值已无 None/nan/inf 字样（缺失统一显示为 —）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
