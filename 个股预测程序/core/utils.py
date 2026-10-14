# -*- coding: utf-8 -*-
"""
通用工具：数值计算、概率函数、HTTP 会话、缓存、A股代码规范化、轻量日志。

本模块只依赖标准库 + numpy（numpy 缺失时有退化实现），可被任何模块安全导入。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pickle
import re
import threading
import time
from typing import Any, Callable, Iterable

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
_LOG_CONFIGURED = False


def get_logger(name: str = "stockai") -> logging.Logger:
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOG_CONFIGURED = True
    return logging.getLogger(name)


log = get_logger()

# --------------------------------------------------------------------------- #
# 数值 / 概率
# --------------------------------------------------------------------------- #
SQRT_2PI = math.sqrt(2.0 * math.pi)


def sigmoid(x: float) -> float:
    """数值稳定的 logistic 函数。"""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float, eps: float = 1e-6) -> float:
    """概率 -> log-odds，自动截断避免除零。"""
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def norm_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """标准正态 CDF（用 math.erf 实现，无需 scipy）。"""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def norm_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    if sigma <= 0:
        return 0.0
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * SQRT_2PI)


def norm_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """正态分布分位数（Acklam 有理逼近，精度 ~1e-9），避免依赖 scipy。"""
    p = min(max(float(p), 1e-9), 1.0 - 1e-9)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        z = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    return mu + sigma * z


def mean(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs: Iterable[float], ddof: int = 1) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(xs)
    if n <= ddof:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def percentile_rank(series, value: float) -> float:
    """value 在 series 中的分位（0~1）。series 可为 list / numpy / pandas Series。"""
    try:
        arr = np.asarray(series, dtype=float) if np is not None else list(series)
        arr = [x for x in arr if x == x]
    except Exception:
        return float("nan")
    if not len(arr):
        return float("nan")
    if np is not None:
        return float((np.asarray(arr) <= value).mean())
    return sum(1 for x in arr if x <= value) / len(arr)


def winsorize(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return lo if x < lo else (hi if x > hi else x)


def tanh_scale(x: float, scale: float = 1.0) -> float:
    """把任意实数压缩到 (-1,1)，常用于把指标值转成方向分。"""
    try:
        return math.tanh(float(x) / (scale if scale else 1.0))
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# A 股代码规范化
# --------------------------------------------------------------------------- #
def normalize_symbol(symbol: str, default_market: str = "cn") -> dict:
    """把各种写法的代码统一成结构化信息。

    支持：``600519`` / ``sh600519`` / ``600519.SH`` / ``SZ000001`` / ``000001.XSHE``
    / ``00700.HK`` / ``AAPL``（美股）。

    Returns
    -------
    dict(code, market, exchange, prefix, secid, yf_symbol, display)
        * ``prefix``    : ``sh`` / ``sz`` / ``bj`` / ``hk`` / ``us``
        * ``secid``     : 东财格式 ``1.600519`` / ``0.000001`` / ``116.00700``
        * ``yf_symbol`` : yfinance 格式 ``600519.SS`` / ``000001.SZ`` / ``0700.HK``
    """
    raw = str(symbol or "").strip().upper().replace(" ", "")
    market = default_market
    code = raw
    prefix = ""

    m = re.match(r"^(SH|SZ|BJ|HK|US)\.?(\w+)$", raw)
    if m:
        prefix = m.group(1).lower()
        code = m.group(2)
    m = re.match(r"^(\w+?)\.(SH|SS|SZ|XSHE|XSHG|BJ|HK|US)$", raw)
    if m:
        code = m.group(1)
        suffix = m.group(2)
        prefix = {"SS": "sh", "SH": "sh", "XSHG": "sh", "SZ": "sz", "XSHE": "sz",
                  "BJ": "bj", "HK": "hk", "US": "us"}[suffix]

    code = re.sub(r"[^0-9A-Z]", "", code)
    if not prefix:
        if re.fullmatch(r"\d{6}", code):
            if code[0] in "56" or code.startswith("11") or code.startswith("5"):
                prefix = "sh"
            elif code[0] in "013" or code.startswith("12"):
                prefix = "sz"
            elif code[0] in "489":
                prefix = "bj"
            else:
                prefix = "sz"
        elif re.fullmatch(r"\d{4,5}", code):
            prefix, market = "hk", "hk"
            code = code.zfill(5)
        else:
            prefix, market = "us", "us"

    if prefix == "hk":
        market = "hk"
    elif prefix == "us":
        market = "us"
    else:
        market = "cn"

    if prefix in ("sh", "sz", "bj"):
        secid = ("1." if prefix == "sh" else "0.") + code
        yf_symbol = code + (".SS" if prefix == "sh" else (".SZ" if prefix == "sz" else ".BJ"))
        display = prefix + code
    elif prefix == "hk":
        secid = "116." + code.zfill(5)
        yf_symbol = code.zfill(4) + ".HK"
        display = "hk" + code.zfill(5)
    else:
        secid = "105." + code
        yf_symbol = code
        display = code

    return {"code": code, "market": market, "exchange": prefix, "prefix": prefix,
            "secid": secid, "yf_symbol": yf_symbol, "display": display,
            "is_cn": market == "cn"}


def board_of(code: str) -> str:
    """判断 A 股所属板块（决定涨跌停幅度）。"""
    code = re.sub(r"\D", "", str(code))
    if code.startswith("688"):
        return "科创板"
    if code.startswith("300") or code.startswith("301"):
        return "创业板"
    if code.startswith(("83", "87", "88", "43", "92")):
        return "北交所"
    if code.startswith(("60", "00", "001", "002", "003")):
        return "主板"
    return "主板"


def limit_up_pct(code: str, name: str = "") -> float:
    """该股票的涨停幅度（%）。"""
    board = board_of(code)
    if "ST" in str(name).upper():
        return 4.8
    return {"科创板": 19.5, "创业板": 19.5, "北交所": 29.5}.get(board, 9.8)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
_SESSION_LOCK = threading.Lock()
_LOCAL = threading.local()


def session() -> "requests.Session":
    """线程内复用的 requests.Session（带重试与默认头）。"""
    if requests is None:
        raise RuntimeError("requests 未安装，无法联网取数")
    s = getattr(_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        from . import config as _cfg  # 延迟导入避免循环
        s.headers.update({
            "User-Agent": _cfg.RUNTIME["user_agent"],
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
        })
        _LOCAL.session = s
    return s


def http_get(url: str, *, params: dict | None = None, headers: dict | None = None,
             timeout: float | None = None, retries: int | None = None,
             encoding: str | None = None, json_body: bool = False) -> Any:
    """带重试与指数退避的 GET。

    返回值：``json_body=True`` 时返回解析后的对象，否则返回文本。
    全部重试失败会抛出最后一次异常——调用方负责降级。
    """
    from . import config as _cfg

    timeout = timeout or _cfg.RUNTIME["http_timeout"]
    retries = _cfg.RUNTIME["http_retries"] if retries is None else retries
    backoff = _cfg.RUNTIME["http_backoff"]
    last_exc: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            r = session().get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code >= 400:
                raise RuntimeError("HTTP %s" % r.status_code)
            if encoding:
                r.encoding = encoding
            if json_body:
                txt = r.text.strip()
                # 兼容 JSONP: cb({...})
                if txt and not txt[0] in "[{":
                    i, j = txt.find("("), txt.rfind(")")
                    if i > 0 and j > i:
                        txt = txt[i + 1:j]
                return json.loads(txt)
            return r.text
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    raise last_exc if last_exc else RuntimeError("http_get failed")


def http_get_many(jobs: list[tuple[str, dict]], timeout: float = 12.0,
                  max_workers: int = 8) -> dict:
    """并发 GET 多个 URL。

    Parameters
    ----------
    jobs : list[(key, {"url":..., "params":..., "json":bool, "headers":...})]

    Returns
    -------
    dict[key] = 响应内容 或 ``None``（失败时，绝不抛异常）
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out: dict[str, Any] = {}

    def _one(job):
        key, spec = job
        try:
            return key, http_get(spec["url"], params=spec.get("params"),
                                 headers=spec.get("headers"), timeout=timeout,
                                 json_body=spec.get("json", False))
        except Exception as exc:  # noqa: BLE001
            log.debug("http_get_many %s 失败: %s", key, exc)
            return key, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_one, j) for j in jobs]
        for fu in as_completed(futs):
            try:
                k, v = fu.result()
                out[k] = v
            except Exception:  # noqa: BLE001
                continue
    return out


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #
_CACHE_MEM: dict[str, tuple[float, Any]] = {}
_MEM_LOCK = threading.Lock()


def cache_key(*parts) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:20]


def cache_get(key: str, ttl: float) -> Any:
    """两级缓存（内存 + 磁盘）读取，未命中返回 None。"""
    now = time.time()
    with _MEM_LOCK:
        hit = _CACHE_MEM.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    try:
        from . import config as _cfg
        path = os.path.join(_cfg.ensure_cache_dir(), key + ".pkl")
        if os.path.exists(path) and now - os.path.getmtime(path) < ttl:
            with open(path, "rb") as fh:
                val = pickle.load(fh)
            with _MEM_LOCK:
                _CACHE_MEM[key] = (now, val)
            return val
    except Exception:  # noqa: BLE001
        pass
    return None


def cache_put(key: str, value: Any) -> None:
    with _MEM_LOCK:
        _CACHE_MEM[key] = (time.time(), value)
    try:
        from . import config as _cfg
        path = os.path.join(_cfg.ensure_cache_dir(), key + ".pkl")
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(value, fh, protocol=4)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        pass


def cached(key: str, ttl: float, producer: Callable[[], Any]) -> Any:
    """读缓存；未命中则调用 producer 并写缓存。producer 抛异常时向上传播。"""
    val = cache_get(key, ttl)
    if val is not None:
        return val
    val = producer()
    if val is not None:
        cache_put(key, val)
    return val


def cache_clear() -> int:
    """清空缓存，返回删除的文件数。"""
    from . import config as _cfg
    n = 0
    with _MEM_LOCK:
        _CACHE_MEM.clear()
    d = _cfg.ensure_cache_dir()
    for fn in os.listdir(d):
        if fn.endswith((".pkl", ".tmp")):
            try:
                os.remove(os.path.join(d, fn))
                n += 1
            except OSError:
                pass
    return n


# --------------------------------------------------------------------------- #
# 其他
# --------------------------------------------------------------------------- #
def rolling_apply(series, window: int, func: Callable[[list], float]) -> list:
    """纯 Python 滚动窗口（pandas 不可用时的兜底）。"""
    out = []
    buf: list = []
    for v in series:
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        out.append(func(buf) if len(buf) == window else float("nan"))
    return out


def fmt_num(x: float, digits: int = 2, unit: str = "") -> str:
    """人类友好数字格式化（自动亿/万）。"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    ax = abs(x)
    if ax >= 1e12:
        return "%.2f万亿%s" % (x / 1e12, unit)
    if ax >= 1e8:
        return "%.2f亿%s" % (x / 1e8, unit)
    if ax >= 1e4:
        return "%.2f万%s" % (x / 1e4, unit)
    return ("%%.%df%%s" % digits) % (x, unit)


def fmt_pct(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return ("%%.%df%%%%" % digits) % x


def retry_call(func: Callable, attempts: int = 3, delay: float = 0.5,
               default: Any = None, label: str = "") -> Any:
    """重试调用；全部失败返回 default（用于"绝不中断"的取数场景）。"""
    for i in range(max(1, attempts)):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            if i == attempts - 1:
                log.debug("retry_call %s 最终失败: %s", label or getattr(func, "__name__", "?"), exc)
                return default
            time.sleep(delay * (2 ** i))
    return default


if __name__ == "__main__":  # pragma: no cover
    print(sigmoid(0), logit(0.5), norm_cdf(0), norm_ppf(0.975))
    for s in ["600519", "sh600519", "000001.SZ", "00700.HK", "AAPL", "688981", "300750"]:
        print(s, "->", normalize_symbol(s))
    print("board 300750 =", board_of("300750"), "limit =", limit_up_pct("300750"))
    print("fmt:", fmt_num(1.234e8), fmt_pct(3.14159))
    print("utils self-test OK")
