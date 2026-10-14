# -*- coding: utf-8 -*-
"""
字体覆盖校验：确保所有可能被写进图表的中文字符，仓库内置字体里都有字形。

为什么需要这个测试
------------------
这个坑真实发生过两次：
  1. 云端没有中文字体 → 图表画成方块；
  2. 内置字体是按"当时的源码字符集"做的子集，后来新增模块带入新汉字
     （铝/锌/锡/镍/领/醇…），字体里没有这些字形 → 又被画成方块。

第 2 类问题非常隐蔽：本地（Windows 有微软雅黑）完全看不出来，只有云端才暴露。
所以用本测试把它固定住——**新增中文文案后必须重新生成字体**：
    python tools/regen_font.py   （脚本在 .work/ 或按需复制到 tools/）

用法：``python tests/test_font_coverage.py``
"""

from __future__ import annotations

import os
import string
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = os.path.join(ROOT, "core", "assets", "NotoSansSC-subset.otf")
SKIP_DIRS = {"__pycache__", ".git", "assets", "data", "reports", "charts",
             ".learn_selftest", ".verify_learn", ".learn_test", "_chart_selftest",
             "_report_selftest", "tests"}

#: 这些字符不参与校验：
#: * 控制字符（\t\n\r）不会渲染；
#: * emoji / 变体选择符等本就不该出现在 matplotlib 图表里（UI 已改用 Material 图标）。
def _ignorable(ch: str) -> bool:
    cp = ord(ch)
    if cp in (9, 10, 13):
        return True
    if 0x1F000 <= cp <= 0x1FAFF or 0xFE00 <= cp <= 0xFE0F or cp == 0x200D:
        return True
    return False


#: 已知内置字体缺少、但也不会进入图表标签的符号（UI 文本用，不走 matplotlib）
KNOWN_OK_MISSING = {"⁻", "ℹ", "⋮", "✅", "✗"}


def collect_chars() -> set:
    chars = set(string.printable) - set("\x0b\x0c")
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            try:
                txt = open(os.path.join(dirpath, fn), encoding="utf-8").read()
            except Exception:  # noqa: BLE001
                continue
            chars |= {c for c in txt if ord(c) > 0x7E}
    return {c for c in chars if not _ignorable(c)}


def font_cmap(path: str) -> set:
    try:
        from fontTools.ttLib import TTFont
    except Exception:  # noqa: BLE001
        return set()
    f = TTFont(path, lazy=True)
    cmap = set()
    for t in f["cmap"].tables:
        cmap |= set(t.cmap.keys())
    return cmap


def main() -> int:
    if not os.path.exists(FONT):
        print("[失败] 找不到内置字体：%s" % FONT)
        return 1
    chars = collect_chars()
    cmap = font_cmap(FONT)
    if not cmap:
        print("[跳过] 未安装 fontTools，无法读取字体 cmap（不影响程序运行）")
        return 0
    missing = sorted(c for c in chars if ord(c) not in cmap
                     and c not in KNOWN_OK_MISSING)
    print("内置字体：%s" % os.path.basename(FONT))
    print("字体字形数：%d ｜ 源码待覆盖字符：%d" % (len(cmap), len(chars)))
    if missing:
        print("\n[失败] 以下字符在源码里出现、但内置字体没有字形，"
              "会在云端图表中显示成方块（%d 个）：" % len(missing))
        for c in missing[:60]:
            print("   %r  U+%04X" % (c, ord(c)))
        print("\n修复方法：重新生成字体子集（覆盖当前源码全部字符）")
        return 1
    print("\n[通过] 源码中所有非 ASCII 字符都在内置字体覆盖范围内，"
          "云端图表不会出现方块字")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
