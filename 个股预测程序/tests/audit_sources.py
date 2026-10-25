# -*- coding: utf-8 -*-
"""
全面审计：逐个探测所有外部数据源的**实际最新数据日期**。

为什么要把失败原因分类
----------------------
上一版只有 ``FRESH / STALE / EMPTY / FAIL``，``EMPTY`` 太模糊——
它既可能是"数据源真的没数据"，也可能是"**我的调用代码写错了**"。
实际发生过两次误判：
  · ``fetch_nav_history`` 返回 DataFrame，我当 dict 取键 → 报 EMPTY
  · ``fetch_financials`` 找错了模块，日期列还叫「报告期」不是「日期」→ 报 EMPTY
这两次都是**探测代码的错**，却看起来像"数据源不可用"。

所以现在分五种状态，责任明确：
  FRESH        数据新鲜
  STALE        数据过期（上游冻结）
  NO_DATA      数据源确实没数据（调用方式已验证正确）
  SHAPE_ERROR  **拿到了数据但解析不出日期**（类型/列名不符）→ 基本是我的错
  CALL_ERROR   **调用直接抛异常**（模块/函数名/参数不对）→ 基本是我的错

后两类会打印诊断（实际类型、可用列名、异常原文），便于立刻定位。
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from core import freshness as F  # noqa: E402


class NoData(Exception):
    """数据源确实没有数据（已确认调用方式正确）。"""


class ShapeError(Exception):
    """拿到了数据，但类型/列名与预期不符，提取不出日期。"""


ROWS: list = []
CALL_PROBLEMS: list = []

DATE_KEYS = ("报告期", "日期", "交易日期", "月份", "统计时间", "时间",
             "date", "Date", "month")


def extract_latest_date(obj, keys=DATE_KEYS):
    """从 DataFrame / dict / 标量提取最新日期；失败抛 ShapeError 并给诊断。"""
    import pandas as pd

    if obj is None:
        raise NoData("返回 None")
    if isinstance(obj, (str, pd.Timestamp)) or hasattr(obj, "year"):
        d = pd.to_datetime(str(obj)[:10], errors="coerce")
        if d is None or pd.isna(d):
            raise ShapeError("标量无法解析为日期: %r" % (obj,))
        return d
    if isinstance(obj, pd.DataFrame):
        if not len(obj):
            raise NoData("DataFrame 有列无行")
        hit = [c for c in obj.columns if any(str(k) in str(c) for k in keys)]
        if not hit:
            raise ShapeError("DataFrame 无日期列；实际列名=%s"
                             % [str(c) for c in list(obj.columns)[:10]])
        s = pd.to_datetime(obj[hit[0]], errors="coerce").dropna()
        if not len(s):
            raise ShapeError("日期列 %r 全部无法解析；样例=%r"
                             % (hit[0], obj[hit[0]].head(3).tolist()))
        return s.max()
    if isinstance(obj, dict):
        for k in keys:
            if obj.get(k):
                return extract_latest_date(obj[k], keys)
        raise ShapeError("dict 无日期键；实际键=%s" % list(obj.keys())[:10])
    raise ShapeError("不支持的类型 %s（样例 %r）"
                     % (type(obj).__name__, str(obj)[:80]))


def probe(name: str, fn) -> None:
    """执行一个探测并记录状态（区分数据源问题与代码问题）。"""
    import traceback

    try:
        obj = fn()
        date_val = extract_latest_date(obj)
    except NoData as exc:
        ROWS.append((name, "—", "NO_DATA", str(exc)[:70]))
        return
    except ShapeError as exc:
        ROWS.append((name, "—", "SHAPE_ERROR", str(exc)[:70]))
        CALL_PROBLEMS.append((name, "SHAPE_ERROR", str(exc)))
        return
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()
        tail = tb[-2].strip() if len(tb) >= 2 else ""
        ROWS.append((name, "—", "CALL_ERROR",
                     "%s: %s" % (type(exc).__name__, str(exc)[:50])))
        CALL_PROBLEMS.append((name, "CALL_ERROR",
                              "%s: %s\n      %s" % (type(exc).__name__,
                                                    str(exc)[:120], tail)))
        return
    r = F.validate(name, date_val)
    ROWS.append((name, r["date"] or "—", r["status"].upper(), r["note"][:70]))


def main() -> int:
    import datetime as dt

    # ---- 行情 ----
    def _daily():
        from core import datasource as ds
        return ds.fetch_daily("600519", days=60)

    def _index():
        from core import global_market as GM
        return GM.fetch_index_history("sh", 100)

    def _blist():
        from core import market_board as MB
        b = MB.fetch_board_list()
        if not b:
            raise NoData("板块列表为空（东财与新浪均不可用）")
        return dt.date.today()

    def _bhist():
        from core import market_board as MB
        b = MB.fetch_board_list()[:3]
        if not b:
            raise NoData("板块列表为空，无法取历史")
        h = MB.fetch_board_histories(b, days=60)
        if not h:
            raise NoData("板块历史全空（push2his 不可达，已有同花顺备份路径）")
        return list(h.values())[0]

    def _nav():
        # fetch_nav_history 返回 DataFrame(date/nav/acc_nav/…)
        from core import fund as FD
        return FD.fetch_nav_history("510300", pages=2)

    # ---- 资金与估值 ----
    def _margin():
        from core import market_fundamental as MF
        d = MF.margin_balance()
        if not d.get("date"):
            raise NoData(d.get("detail") or "两融无日期")
        return d["date"]

    def _north():
        from core import market_fundamental as MF
        d = MF.northbound_flow()
        if not d.get("date"):
            raise NoData(d.get("detail") or "北向无日期")
        return d["date"]

    def _pe():
        from core import market_fundamental as MF
        d = MF.index_valuation("hs300")
        if not d.get("date"):
            raise NoData(d.get("detail") or "PE 无日期")
        return d["date"]

    # ---- 宏观 ----
    def _macro():
        from core import market_context as MC
        env = MC.macro_environment()
        dts = [v.get("date") for v in (env.get("indicators") or {}).values()
               if v.get("date")]
        if not dts:
            raise NoData("宏观指标全部无日期")
        return max(dts)

    def _accounts():
        from core import market_fundamental as MF
        d = MF.new_accounts()
        if not d.get("date"):
            raise NoData(d.get("detail") or "开户数无日期")
        return d["date"]

    def _financials():
        # 财务在 core.fundamental（不在 datasource），返回 DataFrame，日期列=报告期
        from core import fundamental as FU
        return FU.fetch_financials("600519")

    print("开始审计（约 1~3 分钟）…\n")
    items = (("daily", _daily), ("index_history", _index),
             ("board_list", _blist), ("board_history", _bhist),
             ("fund_nav", _nav), ("margin", _margin), ("north", _north),
             ("index_pe", _pe), ("macro", _macro),
             ("accounts", _accounts), ("financials", _financials))
    for nm, fn in items:
        probe(nm, fn)
        print("  %-16s %-12s %-13s %s" % ROWS[-1])

    ok = [r for r in ROWS if r[2] == "FRESH"]
    stale = [r for r in ROWS if r[2] == "STALE"]
    nodata = [r for r in ROWS if r[2] == "NO_DATA"]
    call_bad = [r for r in ROWS if r[2] in ("SHAPE_ERROR", "CALL_ERROR")]

    print("\n" + "=" * 82)
    print("审计结果：%d 项检查" % len(ROWS))
    print("  新鲜 FRESH        : %d  %s"
          % (len(ok), "、".join(r[0] for r in ok)))
    if stale:
        print("  过期 STALE        : %d  %s"
              % (len(stale), "、".join("%s(%s)" % (r[0], r[1]) for r in stale)))
    if nodata:
        print("  无数据 NO_DATA    : %d  %s"
              % (len(nodata), "、".join(r[0] for r in nodata)))
    if call_bad:
        print("  **我的代码有问题** : %d  %s"
              % (len(call_bad), "、".join(r[0] for r in call_bad)))

    if CALL_PROBLEMS:
        print("\n需要修我的探测/调用代码（不是数据源的问题）：")
        for nm, st, msg in CALL_PROBLEMS:
            print("  · %-16s [%s] %s" % (nm, st, msg))
    if nodata:
        print("\n数据源确实无数据（调用方式已验证正确）：")
        for nm, _d, _st, note in nodata:
            print("  · %-16s %s" % (nm, note))

    snap = F.snapshot()
    print("\n全局快照：%s" % snap["detail"][:150])
    # 有 CALL/SHAPE 错误说明审计脚本自己有问题，必须修
    return 1 if call_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
