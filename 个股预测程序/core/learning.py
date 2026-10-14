# -*- coding: utf-8 -*-
"""
预测档案与自我学习 —— 让程序**随着使用不断校准自己**。

为什么需要它
------------
没有这一层，程序每次运行都是"从零开始的同一个程序"：它给出 60% 的上涨概率，
但你永远不知道它**历史上说 60% 的时候实际对了多少**。这一层补上三件事：

1. **记录（Ledger）**：每次预测都留档——日期、周期、概率、期望幅度、
   当时的因子快照、置信度。
2. **核账（Evaluate）**：周期走完后，用真实价格回填"实际涨跌"，
   算出这次预测是对是错、偏差多大。
3. **学习（Learn）**：用累积的核账结果，逐只股票、逐个周期地校准：
   * 概率校准系数——把"过于自信"或"过于保守"的概率拉回实际水平；
   * 类别权重系数——**哪个维度的因子对这只股票真正有效**
     （例如某只股票基本面因子一直准、情绪因子一直反指，系统会据此调权）；
   * 置信度乘子——历史不准的周期自动降低置信度。

与"过拟合"的边界
----------------
* 样本少时**强力收缩**（至少 20 个已核账样本才生效，系数上限 ±40%）；
* 只做**温和的线性校准**，不改因子逻辑本身；
* 每次学习都留痕（``history``），可随时查看、回滚、清空；
* 校准只作用于"概率与置信度"，不会凭空造出预测能力——
  如果历史证明这套信号无效，它会把置信度压得更低，这是正确行为。

存储
----
* ``data/prediction_ledger.jsonl``：一行一条预测记录（追加写，人类可读）
* ``data/calibration.json``：每只股票每个周期的学习结果
"""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from datetime import datetime

from .contracts import HORIZONS, horizon_spec, now_cn, safe_float
from .utils import get_logger

log = get_logger()

_LOCK = threading.Lock()

#: 学习参数（集中管理，便于审计与调参）
LEARN = {
    "min_samples": 20,          # 至少多少个已核账样本才开始生效
    "full_samples": 120,        # 达到多少个样本时收缩系数为 1（完全生效）
    "max_adjust": 0.40,         # 概率校准的最大调整幅度（对数几率口径）
    "max_confidence_mult": 1.35,  # 置信度乘子上限
    "min_confidence_mult": 0.55,  # 置信度乘子下限
    "clamp_prob": (0.05, 0.95),
    # 回填样本的权重折扣（历史回填缺少"当时的信息环境"，可信度略低）
    "backfill_weight": 0.7,
}


# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
def data_dir() -> str:
    """预测数据目录（默认放在项目下的 data/，可用环境变量覆盖）。"""
    d = os.environ.get("STOCKAI_DATA_DIR")
    if not d:
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:  # noqa: BLE001
        d = os.path.join(os.path.expanduser("~"), ".stock_predictor_data")
        os.makedirs(d, exist_ok=True)
    return d


def ledger_path() -> str:
    return os.path.join(data_dir(), "prediction_ledger.jsonl")


def calibration_path() -> str:
    return os.path.join(data_dir(), "calibration.json")


# --------------------------------------------------------------------------- #
# 一、记录预测
# --------------------------------------------------------------------------- #
def record_prediction(result, *, extra: dict | None = None) -> dict:
    """把一次完整预测（8 个周期）写入档案，返回写入的记录。

    只记录"重新计算可得"的信息 + 当时的关键状态（因子快照），
    便于日后回答"当时为什么这么判、后来对不对"。
    """
    try:
        if result is None or not getattr(result, "forecasts", None):
            return {}
        bundle = getattr(result, "bundle", None)
        as_of = str(getattr(result, "as_of", "") or "")
        if not as_of:
            return {}
        # 同一只股票 + 同一天 + 同周期 只留一条（重复分析时覆盖前一条最有价值）
        run_id = uuid.uuid4().hex[:12]
        snap_ts = now_cn().strftime("%Y-%m-%d %H:%M:%S")
        # 因子快照：只留"有方向"的因子的 id 与带符号强度，控制体积
        factor_snapshot = []
        for f in (getattr(bundle, "factors", []) or []):
            if getattr(f, "is_actionable", False):
                factor_snapshot.append({
                    "id": f.id, "cat": f.category, "dir": f.direction,
                    "s": round(float(f.strength), 3), "r": round(float(f.reliability), 3),
                })
        rows = []
        for fc in result.forecasts:
            rows.append({
                "run_id": run_id,
                "recorded_at": snap_ts,
                "as_of": as_of,                     # 预测所依据的数据截止日
                "symbol": str(getattr(result, "symbol", "") or ""),
                "name": str(getattr(result, "name", "") or ""),
                "price": safe_float(getattr(result, "price", None)),
                "horizon": fc.horizon,
                "label": fc.label,
                "bars": fc.bars,
                "p_up": round(float(fc.p_up), 4),
                "p_up_rule": round(float(getattr(fc, "_p_rule", fc.p_up) or fc.p_up), 4),
                "exp_return": round(float(fc.exp_return), 4),
                "ci_low": round(float(fc.ci_low), 4),
                "ci_high": round(float(fc.ci_high), 4),
                "sigma": round(float(fc.sigma), 4),
                "confidence": round(float(fc.confidence), 4),
                "trend": fc.trend,
                "strength_score": round(float(fc.strength_score), 3),
                "model_mix": dict(fc.model_mix or {}),
                "drivers": [d.get("因子") for d in (fc.drivers or [])][:8],
                "counter": [d.get("因子") for d in (fc.counter or [])][:6],
                "factors": factor_snapshot if fc.horizon == "d20" else [],
                "n_factors": len(getattr(bundle, "factors", []) or []) if bundle else 0,
                "evaluated": False,
                **(extra or {}),
            })
        _append(rows)
        log.info("已记录 %s(%s) 的 %d 条预测（依据数据日 %s）",
                 getattr(result, "name", ""), getattr(result, "symbol", ""),
                 len(rows), as_of)
        return {"run_id": run_id, "rows": len(rows),
                "path": ledger_path()}
    except Exception as exc:  # noqa: BLE001
        log.warning("记录预测失败：%s", exc)
        return {}


def _append(rows: list) -> None:
    if not rows:
        return
    with _LOCK:
        with open(ledger_path(), "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def load_ledger(symbol: str | None = None, *, evaluated: bool | None = None,
                limit: int | None = None) -> list:
    """读取档案。``symbol`` 为 None 时返回全部。"""
    path = ledger_path()
    if not os.path.exists(path):
        return []
    out: list = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if symbol and str(r.get("symbol")) != str(symbol):
                    continue
                if evaluated is not None and bool(r.get("evaluated")) != evaluated:
                    continue
                out.append(r)
    except Exception as exc:  # noqa: BLE001
        log.warning("读取预测档案失败：%s", exc)
        return []
    # 同一天同一周期只保留最后一条（覆盖早期重复分析）
    dedup: dict = {}
    for r in out:
        dedup[(r.get("symbol"), r.get("as_of"), r.get("horizon"))] = r
    rows = list(dedup.values())
    rows.sort(key=lambda r: (str(r.get("as_of")), str(r.get("horizon"))))
    if limit:
        rows = rows[-limit:]
    return rows


def _rewrite(rows: list) -> None:
    """重写整个档案（用于回填核账结果）。"""
    tmp = ledger_path() + ".tmp"
    with _LOCK:
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, ledger_path())


# --------------------------------------------------------------------------- #
# 二、核账（用真实价格回填结果）
# --------------------------------------------------------------------------- #
def evaluate_pending(symbol: str, daily: "object" = None) -> dict:
    """把已经走完周期的预测与真实价格比对，回填核账结果。

    Parameters
    ----------
    symbol : 股票代码
    daily : 该股日线 DataFrame（含 date/close）；为 None 时自动取数

    Returns
    -------
    dict：``{"evaluated": n, "pending": m, "detail": str}``
    """
    import pandas as pd

    rows = load_ledger(symbol)
    if not rows:
        return {"evaluated": 0, "pending": 0, "detail": "档案为空"}
    if daily is None or not hasattr(daily, "columns"):
        try:
            from . import datasource as ds

            daily = ds.fetch_daily(symbol, days=1600)
        except Exception as exc:  # noqa: BLE001
            return {"evaluated": 0, "pending": 0, "detail": "取数失败：%s" % exc}
    if daily is None or not len(daily):
        return {"evaluated": 0, "pending": 0, "detail": "无行情数据"}

    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    dates = list(d["date"])
    closes = [float(x) for x in d["close"]]

    def _price_on_or_after(ts):
        for i, dt in enumerate(dates):
            if dt >= ts:
                return closes[i], dt
        return None, None

    n_eval = 0
    n_pending = 0
    for r in rows:
        if r.get("evaluated"):
            continue
        try:
            as_of = pd.to_datetime(r.get("as_of"))
            spec = horizon_spec(str(r.get("horizon")))
        except Exception:  # noqa: BLE001
            n_pending += 1
            continue
        # 目标日 = 依据日 + bars 个交易日 → 用自然日近似后再对齐到实际交易日
        target = as_of + pd.Timedelta(days=int(spec.days * 1.6) + 4)
        last_dt = dates[-1] if dates else None
        if last_dt is None or last_dt < target:
            n_pending += 1                      # 周期还没走完
            continue
        # 找"依据日之后第 bars 个交易日"
        try:
            start_idx = next(i for i, dt in enumerate(dates) if dt >= as_of)
        except StopIteration:
            n_pending += 1
            continue
        end_idx = start_idx + int(spec.bars)
        if end_idx >= len(dates):
            n_pending += 1
            continue
        p0 = closes[start_idx]
        p1 = closes[end_idx]
        if not p0 or p0 != p0:
            n_pending += 1
            continue
        actual = (p1 / p0 - 1) * 100
        p_up = safe_float(r.get("p_up"))
        exp_ret = safe_float(r.get("exp_return"))
        spec_thr = 5.0
        try:
            from . import config as _cfg
            spec_thr = float((_cfg.FORECAST.get("big_move_pct") or {}).get(
                str(r.get("horizon")), 5.0))
        except Exception:  # noqa: BLE001
            pass
        r.update({
            "evaluated": True,
            "eval_date": str(dates[end_idx])[:10],
            "eval_price": round(p1, 4),
            "actual_return": round(actual, 4),
            "actual_up": bool(actual > 0),
            "hit": bool((p_up >= 0.5) == (actual > 0)) if p_up == p_up else None,
            "direction_error": (round(float(p_up) - (1.0 if actual > 0 else 0.0), 4)
                                if p_up == p_up else None),
            "return_error": (round(float(exp_ret) - actual, 4)
                             if exp_ret == exp_ret else None),
            # 大涨/大跌方向是否也叫对（用配置里该周期的阈值）
            "big_move": bool(abs(actual) >= spec_thr),
            "big_hit": (bool((actual >= spec_thr) if p_up >= 0.5
                             else (actual <= -spec_thr)) if p_up == p_up else None),
        })
        n_eval += 1
    if n_eval:
        _rewrite(rows)
    detail = ("本次核账 %d 条，尚有 %d 条周期未走完" % (n_eval, n_pending)
              if n_eval else "暂无可核账的记录（%d 条周期未走完）" % n_pending)
    return {"evaluated": n_eval, "pending": n_pending, "detail": detail}


# --------------------------------------------------------------------------- #
# 三之零、历史回填 —— 解决"冷启动"问题
# --------------------------------------------------------------------------- #
def backfill_history(symbol: str, daily, *, step: int = 5, max_points: int = 60,
                     name: str = "") -> dict:
    """用**历史数据回填虚拟预测**，让自我学习第一天就有样本可用。

    为什么需要
    ----------
    真实档案必须等周期走完才能核账——短周期几天、长周期要一年。
    如果只靠"边用边学"，新用户前几个月完全没有校准，等于没有学习功能。

    做法
    ----
    在历史序列上滚动取"过去某一天"作为预测基准日，用**当时可见的数据**
    （严格因果：只用该日及之前）算出规则分与概率，再标注该日已实现的
    真实收益。这些记录与真实预测**同构**（``source="backfill"``），
    因此可以走同一套核账与学习流程。

    注意：回填样本来自历史，与真实预测相比少了"当时的信息环境"，
    所以学习时会对它们打一个折扣（见 :func:`learn`）。
    """
    import numpy as np
    import pandas as pd

    from .contracts import AnalysisBundle
    from .forecast import cross_horizon_consistency, estimate_volatility, fast_rule_score_history

    if daily is None or not hasattr(daily, "columns") or len(daily) < 120:
        return {"added": 0, "detail": "历史数据不足（需 ≥120 根）"}
    _init_scale()
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    n = len(d)
    closes = d["close"].astype(float).to_numpy()
    vol = estimate_volatility(d)

    # 先算好每个周期的历史规则分（严格因果）
    scores = {}
    for h in HORIZONS:
        if n > h.bars + 60:
            try:
                scores[h.key] = fast_rule_score_history(d, h.key)
            except Exception:  # noqa: BLE001
                continue
    if not scores:
        return {"added": 0, "detail": "无法计算历史规则分"}

    # 选基准日：从"最早能算分"到"该周期已走完"的区间内均匀取点
    shortest = min(horizon_spec(k).bars for k in scores)
    idxs = list(range(70, n - shortest - 1, max(1, int(step))))
    if len(idxs) > max_points:
        pick = np.linspace(0, len(idxs) - 1, max_points).astype(int)
        idxs = [idxs[i] for i in pick]

    rows: list = []
    for i in idxs:
        as_of = str(d["date"].iloc[i])[:10]
        price = float(closes[i])
        if not price or price != price:
            continue
        for h in HORIZONS:
            sc = scores.get(h.key)
            if sc is None or i >= n - h.bars:
                continue
            s = float(sc[i]) if np.isfinite(sc[i]) else 0.0
            # 与真实预测同口径：logit 概率（不含 AI 融合与学习校准）
            k = float((LEARN_PROB_SCALE or {}).get(h.key, 1.2))
            conf = 0.55
            p_up = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, k * s * conf))))
            p_up = min(max(p_up, 0.05), 0.95)
            sigma = max(vol.get("daily_pct") or 2.0, 0.3) * math.sqrt(max(h.bars, 1)) * 0.92
            try:
                z = _norm_ppf(p_up)
            except Exception:  # noqa: BLE001
                z = 0.0
            exp_ret = z * sigma * 0.45
            actual = (closes[i + h.bars] / price - 1) * 100
            rows.append({
                "run_id": "backfill",
                "recorded_at": now_cn().strftime("%Y-%m-%d %H:%M:%S"),
                "as_of": as_of, "symbol": str(symbol), "name": name or str(symbol),
                "price": round(price, 4),
                "horizon": h.key, "label": h.label, "bars": h.bars,
                "p_up": round(p_up, 4), "p_up_rule": round(p_up, 4),
                "exp_return": round(exp_ret, 4),
                "ci_low": round(exp_ret - 0.9 * sigma, 4),
                "ci_high": round(exp_ret + 0.9 * sigma, 4),
                "sigma": round(sigma, 4), "confidence": conf,
                "trend": "看涨" if p_up > 0.55 else ("看跌" if p_up < 0.45 else "震荡"),
                "strength_score": round(s * 100, 3), "model_mix": {},
                "drivers": [], "counter": [], "factors": [],
                "n_factors": 0,
                "source": "backfill",
                # 直接带上核账结果（历史已经实现）
                "evaluated": True,
                "eval_date": str(d["date"].iloc[i + h.bars])[:10],
                "eval_price": round(float(closes[i + h.bars]), 4),
                "actual_return": round(actual, 4),
                "actual_up": bool(actual > 0),
                "hit": bool((p_up >= 0.5) == (actual > 0)),
                "direction_error": round(p_up - (1.0 if actual > 0 else 0.0), 4),
                "return_error": round(exp_ret - actual, 4),
                "big_move": bool(abs(actual) >= 5.0),
                "big_hit": bool((actual >= 5.0) if p_up >= 0.5 else (actual <= -5.0)),
            })
    if not rows:
        return {"added": 0, "detail": "未能生成回填样本"}

    # 与已有档案合并：回填样本按 (symbol, as_of, horizon) 去重，不覆盖真实记录
    existing = load_ledger()
    keys_existing = {(str(r.get("symbol")), str(r.get("as_of")), str(r.get("horizon")))
                     for r in existing}
    fresh = [r for r in rows
             if (r["symbol"], r["as_of"], r["horizon"]) not in keys_existing]
    if fresh:
        _append(fresh)
    return {"added": len(fresh), "total_generated": len(rows),
            "detail": "已回填 %d 条历史样本（滚动 %d 个基准日，周期走完的才保留）"
                      % (len(fresh), len(idxs))}


#: 供回填使用的周期敏感度（与 config.FORECAST 保持一致）
LEARN_PROB_SCALE: dict = {}


def _norm_ppf(p: float) -> float:
    try:
        from .utils import norm_ppf

        return float(norm_ppf(p))
    except Exception:  # noqa: BLE001
        return 0.0


def _init_scale() -> None:
    global LEARN_PROB_SCALE
    if LEARN_PROB_SCALE:
        return
    try:
        from . import config as _cfg

        LEARN_PROB_SCALE = dict(_cfg.FORECAST.get("score_to_prob_scale") or {})
    except Exception:  # noqa: BLE001
        LEARN_PROB_SCALE = {}


# --------------------------------------------------------------------------- #
# 三、学习（用核账结果校准）
# --------------------------------------------------------------------------- #
def _shrink(n: int) -> float:
    """样本量→收缩系数（0~1）。样本越多越敢下手，避免小样本过拟合。"""
    if n < LEARN["min_samples"]:
        return 0.0
    if n >= LEARN["full_samples"]:
        return 1.0
    return (n - LEARN["min_samples"]) / float(LEARN["full_samples"] - LEARN["min_samples"])


def learn(symbol: str | None = None, *, save: bool = True) -> dict:
    """用已核账样本学习校准系数。

    回填样本（``source == "backfill"``）会按 :data:`LEARN` 的 ``backfill_weight``
    打折计入——它们来自历史、缺少"当时的信息环境"，可信度略低于真实预测。
    """
    rows = [r for r in load_ledger(symbol) if r.get("evaluated")]
    if not rows:
        return {"by_symbol": {}, "overall": {}, "samples": 0,
                "detail": "尚无已核账样本，学习未启动（需要先有预测、且周期走完）"}

    # 按 股票 × 周期 分组
    groups: dict = {}
    for r in rows:
        groups.setdefault((str(r.get("symbol")), str(r.get("horizon"))), []).append(r)

    by_symbol: dict = {}
    for (sym, hz), items in groups.items():
        # 有效样本量：真实记录算 1，回填记录按权重折扣
        n = len(items)
        n_eff = sum(LEARN["backfill_weight"] if str(x.get("source")) == "backfill"
                    else 1.0 for x in items)
        n_backfill = sum(1 for x in items if str(x.get("source")) == "backfill")
        hits = [bool(x.get("hit")) for x in items if x.get("hit") is not None]
        hit_rate = (sum(hits) / len(hits)) if hits else float("nan")
        # 平均概率与实际胜率的偏差 → 对数几率校准
        p_avg = sum(safe_float(x.get("p_up")) for x in items
                    if safe_float(x.get("p_up")) == safe_float(x.get("p_up")))
        p_avg_n = sum(1 for x in items
                      if safe_float(x.get("p_up")) == safe_float(x.get("p_up")))
        p_avg = (p_avg / p_avg_n) if p_avg_n else float("nan")
        # Brier 分数（越低越好）
        briers = []
        for x in items:
            p = safe_float(x.get("p_up"))
            if p != p:
                continue
            y = 1.0 if x.get("actual_up") else 0.0
            briers.append((p - y) ** 2)
        brier = sum(briers) / len(briers) if briers else float("nan")
        # 幅度误差（期望收益 vs 实际）
        errs = [abs(safe_float(x.get("return_error"))) for x in items
                if safe_float(x.get("return_error")) == safe_float(x.get("return_error"))]
        mae = sum(errs) / len(errs) if errs else float("nan")
        # 多空各自的平均实际收益（判断方向信号是否真有价值）
        up_rets = [safe_float(x.get("actual_return")) for x in items
                   if safe_float(x.get("p_up")) >= 0.5
                   and safe_float(x.get("actual_return")) == safe_float(x.get("actual_return"))]
        dn_rets = [safe_float(x.get("actual_return")) for x in items
                   if safe_float(x.get("p_up")) < 0.5
                   and safe_float(x.get("actual_return")) == safe_float(x.get("actual_return"))]

        shrink = _shrink(int(round(n_eff)))
        # 校准量：logit(p_avg) - logit(hit_rate)，负值表示"过于自信"
        adj = 0.0
        if shrink > 0 and p_avg == p_avg and hit_rate == hit_rate:
            try:
                p_c = min(max(p_avg, 1e-4), 1 - 1e-4)
                h_c = min(max(hit_rate, 1e-4), 1 - 1e-4)
                raw_adj = math.log(h_c / (1 - h_c)) - math.log(p_c / (1 - p_c))
                adj = max(-LEARN["max_adjust"], min(LEARN["max_adjust"], raw_adj)) * shrink
            except Exception:  # noqa: BLE001
                adj = 0.0
        # 置信度乘子：命中率明显高于/低于 50% 时放大/缩小
        conf_mult = 1.0
        if shrink > 0 and hit_rate == hit_rate:
            conf_mult = 1.0 + (hit_rate - 0.5) * 2.0 * shrink
            conf_mult = max(LEARN["min_confidence_mult"],
                            min(LEARN["max_confidence_mult"], conf_mult))

        by_symbol.setdefault(sym, {})[hz] = {
            "samples": n,
            "samples_effective": round(n_eff, 1),
            "samples_backfill": n_backfill,
            "hit_rate": round(hit_rate, 4) if hit_rate == hit_rate else None,
            "p_avg": round(p_avg, 4) if p_avg == p_avg else None,
            "brier": round(brier, 4) if brier == brier else None,
            "mae": round(mae, 3) if mae == mae else None,
            "avg_ret_when_bull": round(sum(up_rets) / len(up_rets), 3) if up_rets else None,
            "avg_ret_when_bear": round(sum(dn_rets) / len(dn_rets), 3) if dn_rets else None,
            "shrink": round(shrink, 3),
            "prob_adjust_logit": round(adj, 4),
            "confidence_mult": round(conf_mult, 4),
        }

    # 类别权重学习：哪个维度的因子真正有效（用 d20 样本、按因子快照分组）
    dim_stats = _learn_dimension_weights([r for r in rows if r.get("horizon") == "d20"])

    payload = {
        "updated_at": now_cn().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(rows),
        "by_symbol": by_symbol,
        "dimension_weights": dim_stats.get("weights", {}),
        "dimension_detail": dim_stats.get("detail", []),
    }
    if save:
        try:
            hist = _load_calibration()
            prev = hist.get("history", [])
            prev.append({"updated_at": payload["updated_at"],
                         "total_samples": len(rows)})
            hist["history"] = prev[-20:]
            hist["latest"] = payload
            with open(calibration_path(), "w", encoding="utf-8") as fh:
                json.dump(hist, fh, ensure_ascii=False, indent=2, default=str)
        except Exception as exc:  # noqa: BLE001
            log.warning("保存校准结果失败：%s", exc)
    return {**payload, "samples": len(rows),
            "detail": "已用 %d 条核账样本更新校准（%d 只股票）"
                      % (len(rows), len(by_symbol))}


def _learn_dimension_weights(rows: list) -> dict:
    """按因子类别统计"出现该类别因子时方向是否更准"，产出权重系数。

    做法：对每个类别，比较"含该类因子且看多"的样本中实际上涨的比例，
    与整体基准率之差 → 转换成 0.7~1.3 的权重系数（温和调整）。
    """
    if len(rows) < LEARN["min_samples"]:
        return {"weights": {}, "detail": []}
    base_up = sum(1 for r in rows if r.get("actual_up")) / len(rows)
    cats: dict = {}
    for r in rows:
        for f in (r.get("factors") or []):
            c = f.get("cat")
            if not c:
                continue
            key = (c, int(f.get("dir", 0)))
            cats.setdefault(key, []).append(bool(r.get("actual_up")))
    detail = []
    weights: dict = {}
    for (cat, direction), ups in cats.items():
        n = len(ups)
        if n < 10 or direction == 0:
            continue
        rate = sum(ups) / n
        # 该因子方向为看涨时：命中率高于基准 → 该类别有效
        edge = rate - base_up
        if direction < 0:
            edge = -edge
        w = 1.0 + max(-0.3, min(0.3, edge * 1.5))
        weights[cat] = round(w, 3)
        detail.append({"category": cat, "direction": direction, "samples": n,
                       "hit_rate": round(rate, 4), "base_rate": round(base_up, 4),
                       "edge": round(edge, 4), "weight": round(w, 3)})
    return {"weights": weights, "detail": detail}


def _load_calibration() -> dict:
    try:
        if os.path.exists(calibration_path()):
            with open(calibration_path(), encoding="utf-8") as fh:
                return json.load(fh) or {}
    except Exception as exc:  # noqa: BLE001
        log.debug("读取校准文件失败：%s", exc)
    return {}


def load_calibration() -> dict:
    """读回最近一次学习结果。"""
    return (_load_calibration().get("latest") or {})


def get_adjustments(symbol: str, horizon: str) -> dict:
    """取某股票某周期的校准参数（供预测引擎调用）。"""
    cal = load_calibration()
    node = ((cal.get("by_symbol") or {}).get(str(symbol)) or {}).get(str(horizon))
    if not node:
        return {}
    return node


def get_dimension_weights(symbol: str | None = None) -> dict:
    """取学习到的类别权重系数（供因子汇总时使用）。"""
    cal = load_calibration()
    return cal.get("dimension_weights") or {}


def calibration_report(symbol: str | None = None) -> dict:
    """生成人类可读的学习报告（供界面/命令行展示）。"""
    rows_all = load_ledger(symbol)
    rows_ok = [r for r in rows_all if r.get("evaluated")]
    cal = load_calibration()
    by_sym = (cal.get("by_symbol") or {})
    if symbol and symbol in by_sym:
        by_sym = {symbol: by_sym[symbol]}
    table = []
    for sym, hzs in by_sym.items():
        for hz, v in hzs.items():
            try:
                lab = horizon_spec(hz).label
            except Exception:  # noqa: BLE001
                lab = hz
            table.append({
                "代码": sym, "周期": lab, "已核账样本": v.get("samples"),
                "有效样本": v.get("samples_effective"),
                "其中回填": v.get("samples_backfill"),
                "方向命中率": ("%.1f%%" % (v["hit_rate"] * 100)
                              if v.get("hit_rate") is not None else "—"),
                "平均预测概率": ("%.1f%%" % (v["p_avg"] * 100)
                                if v.get("p_avg") is not None else "—"),
                "Brier分数": v.get("brier"), "幅度平均误差%": v.get("mae"),
                "看多时均收益%": v.get("avg_ret_when_bull"),
                "看空时均收益%": v.get("avg_ret_when_bear"),
                "概率校准": v.get("prob_adjust_logit"),
                "置信度乘子": v.get("confidence_mult"),
                "生效程度": v.get("shrink"),
            })
    return {
        "total_records": len(rows_all),
        "evaluated": len(rows_ok),
        "pending": len(rows_all) - len(rows_ok),
        "last_update": cal.get("updated_at"),
        "dimension_weights": cal.get("dimension_weights") or {},
        "dimension_detail": cal.get("dimension_detail") or [],
        "table": table,
        "detail": ("共 %d 条预测记录，其中 %d 条已核账、%d 条周期未走完；"
                   "上次学习：%s" % (len(rows_all), len(rows_ok),
                                     len(rows_all) - len(rows_ok),
                                     cal.get("updated_at") or "尚未学习")),
    }


def reset(what: str = "calibration") -> str:
    """清空学习结果（``calibration``）或连同档案一起清空（``all``）。"""
    try:
        if what in ("calibration", "all") and os.path.exists(calibration_path()):
            os.remove(calibration_path())
        if what == "all" and os.path.exists(ledger_path()):
            os.remove(ledger_path())
        return "已清空：%s" % ("校准结果与预测档案" if what == "all" else "校准结果")
    except Exception as exc:  # noqa: BLE001
        return "清空失败：%s" % exc


# --------------------------------------------------------------------------- #
# 四、主入口：跑一轮"核账 → 学习"
# --------------------------------------------------------------------------- #
def run_learning_cycle(symbol: str, daily=None, *, save: bool = True) -> dict:
    """对某只股票执行一轮完整的学习闭环。"""
    ev = evaluate_pending(symbol, daily)
    lr = learn(symbol, save=save)
    return {"evaluate": ev, "learn": lr,
            "detail": "%s；%s" % (ev.get("detail"), lr.get("detail"))}


if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")

    import pandas as pd

    from .contracts import AnalysisBundle, ProbabilityForecast, PredictionResult

    # 用项目内的临时目录，避免污染真实档案（系统 temp 在受限环境下可能不可写）
    os.environ["STOCKAI_DATA_DIR"] = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".learn_selftest")
    print("测试数据目录：", data_dir())

    # ---- 构造 60 天前的一次预测（d1 周期，今天已可核账）----
    import numpy as np

    n = 200
    dates = pd.bdate_range("2026-01-01", periods=n)
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0015, 0.02, n)          # 轻微上行漂移
    closes = 20 * np.cumprod(1 + rets)
    daily = pd.DataFrame({"date": dates, "open": closes * 0.995, "high": closes * 1.01,
                          "low": closes * 0.99, "close": closes,
                          "volume": rng.integers(1e6, 4e6, n).astype(float)})

    as_of = str(dates[5])[:10]
    for i in range(6):
        fc = [ProbabilityForecast(horizon=h, label=horizon_spec(h).label,
                                  p_up=0.60, exp_return=1.0, ci_low=-2.0, ci_high=4.0,
                                  sigma=2.0, confidence=0.6, strength_score=20.0)
              for h in ("d1", "d5")]
        res = PredictionResult(symbol="TEST", name="测试股", as_of=as_of,
                               price=float(closes[5]), forecasts=fc,
                               bundle=AnalysisBundle(symbol="TEST", name="测试股",
                                                     as_of=as_of))
        record_prediction(res)
        break

    print("\n档案记录数：", len(load_ledger("TEST")))
    ev = evaluate_pending("TEST", daily)
    print("核账：", ev)
    lr = learn("TEST")
    print("学习：", lr.get("detail"))
    rep = calibration_report("TEST")
    print("\n学习报告：")
    print("  ", rep["detail"])
    for row in rep["table"]:
        print("   %s %s 样本=%s 命中率=%s 平均预测概率=%s Brier=%s 置信度乘子=%s"
              % (row["代码"], row["周期"], row["已核账样本"], row["方向命中率"],
                 row["平均预测概率"], row["Brier分数"], row["置信度乘子"]))

    # 断言：记录 → 核账 → 学习 链路打通
    assert len(load_ledger("TEST")) == 2, "应记录 2 条（d1/d5）"
    assert ev["evaluated"] >= 1, "d1 应可核账"
    assert lr.get("samples", 0) >= 1, "应有已核账样本"
    # 同一天重复记录应去重
    rows = load_ledger("TEST")
    assert len({(r["symbol"], r["as_of"], r["horizon"]) for r in rows}) == len(rows), \
        "同一天同周期应去重"
    print("\nreset:", reset("all"))
    print("清空后档案数：", len(load_ledger("TEST")))
    print("\nlearning self-test OK")
