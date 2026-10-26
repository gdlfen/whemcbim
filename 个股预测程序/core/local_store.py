# -*- coding: utf-8 -*-
"""
本地私有数据存储：**AI 配置与手动宏观数据只在操作设备上保存使用**。

为什么需要这一层
----------------
原来这两个文件直接写到 ``data/`` 目录：
  · ``data/ai_config.json``     —— 含 API Key
  · ``data/macro_manual.json``  —— 用户手工填的宏观值

本地运行时没问题（设备就是本机）。但在 **Streamlit Cloud 上运行**时，
``data/`` 在服务器容器里：
  1. **所有访问者共享同一个文件** —— 一个人填的 Key，其他人也在用，
     属于凭据泄露；手动宏观值也会互相覆盖。
  2. 容器重建（重新部署/休眠唤醒）后文件丢失，填了也白填。

所以这里按部署环境分流：
  · **本地部署**（路径不是 /mount/src、且有 STREAMLIT_CLOUD 未设置）
    → 落盘到 ``data/``，重启后仍在（设备级持久化）
  · **云端部署**
    → **绝不落盘**，改为按「浏览器会话 ID」隔离的内存存储
      （每个访客自己一份，互不可见；刷新即失效，界面会提示）

界面用 :func:`describe` 告知用户当前存在哪里，避免误解。
"""
from __future__ import annotations

import json
import os
import threading

from .utils import get_logger

log = get_logger()

_LOCK = threading.RLock()

#: 云端模式下的内存存储：{session_id: {name: data}}
_MEM: dict = {}


def is_local() -> bool:
    """当前是否本地部署（数据可安全落盘到本机）。"""
    if os.environ.get("STREAMLIT_CLOUD") or os.environ.get("IS_STREAMLIT_CLOUD"):
        return False
    try:
        p = os.path.abspath(__file__).replace("\\", "/")
        if p.startswith("/mount/src"):        # Streamlit Cloud 的挂载路径
            return False
    except Exception:  # noqa: BLE001
        pass
    return True


def _session_id() -> str:
    """当前浏览器会话 ID（云端用于隔离不同访客）。"""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is not None and getattr(ctx, "session_id", None):
            return str(ctx.session_id)
    except Exception:  # noqa: BLE001
        pass
    return "__process__"


def _base_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data")


def path_of(name: str) -> str:
    return os.path.join(_base_dir(), name)


def load(name: str) -> dict:
    """读取私有数据；云端从会话内存读，本地从文件读。"""
    if not is_local():
        with _LOCK:
            return dict(_MEM.get(_session_id(), {}).get(name) or {})
    p = path_of(name)
    try:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
            return d if isinstance(d, dict) else {}
    except Exception as exc:  # noqa: BLE001
        log.debug("读私有数据失败 %s: %s", name, exc)
    return {}


def save(name: str, data: dict) -> bool:
    """写私有数据；**云端不落盘**，只存当前会话内存。"""
    if not is_local():
        with _LOCK:
            sid = _session_id()
            box = _MEM.setdefault(sid, {})
            box[name] = dict(data or {})
        return True
    try:
        os.makedirs(_base_dir(), exist_ok=True)
        p = path_of(name)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(data or {}, fh, ensure_ascii=False, indent=2)
        try:
            os.chmod(p, 0o600)                # 仅本用户可读写
        except Exception:  # noqa: BLE001
            pass
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("写私有数据失败 %s: %s", name, exc)
        return False


def describe(name: str) -> str:
    """给界面用的存储位置说明。"""
    if is_local():
        return "仅保存在本机：%s（不会上传到任何地方）" % path_of(name)
    return ("**仅保存在当前浏览器会话**（云端部署不落盘）："
            "刷新页面后需重新填写。如需长期保存，请把程序跑在自己的电脑上。")


def ensure_gitignored() -> None:
    """确保私有文件被 .gitignore 排除（本地部署时防止误提交）。"""
    try:
        gi = os.path.join(os.path.dirname(_base_dir()), ".gitignore")
        if not os.path.exists(gi):
            return
        txt = open(gi, encoding="utf-8").read()
        need = [n for n in ("data/ai_config.json", "data/macro_manual.json")
                if n not in txt]
        if need:
            with open(gi, "a", encoding="utf-8") as fh:
                fh.write("\n# 设备私有数据，禁止提交\n")
                for n in need:
                    fh.write(n + "\n")
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")

    print("=== 1. 环境判定 ===")
    print("  is_local:", is_local())
    print("  describe:", describe("ai_config.json")[:80])

    print("\n=== 2. 读写往返 ===")
    save("__selftest__.json", {"k": 1, "中文": "测试"})
    d = load("__selftest__.json")
    print("  读回:", d)
    assert d.get("k") == 1 and d.get("中文") == "测试"

    print("\n=== 3. 云端模式不落盘 ===")
    _orig = os.environ.get("STREAMLIT_CLOUD")
    os.environ["STREAMLIT_CLOUD"] = "1"
    try:
        assert is_local() is False
        p = path_of("__cloud_probe__.json")
        if os.path.exists(p):
            os.remove(p)
        save("__cloud_probe__.json", {"a": 2})
        assert not os.path.exists(p), "云端模式不应写文件"
        assert load("__cloud_probe__.json").get("a") == 2, "应能从会话内存读回"
        print("  云端：未生成文件，内存可读回 ✓")
    finally:
        if _orig is None:
            os.environ.pop("STREAMLIT_CLOUD", None)
        else:
            os.environ["STREAMLIT_CLOUD"] = _orig

    print("\n=== 4. 清理自测文件 ===")
    try:
        p = path_of("__selftest__.json")
        if os.path.exists(p):
            os.remove(p)
        print("  已清理")
    except Exception as exc:  # noqa: BLE001
        print("  清理失败（可忽略）:", str(exc)[:40])

    print("\n=== 5. gitignore 覆盖 ===")
    ensure_gitignored()
    print("  已确保 data/ai_config.json 与 data/macro_manual.json 被忽略")

    print("\nlocal_store self-test OK")
