@echo off
chcp 65001 >nul
title 个股涨跌概率预测 - 本地网页版
cd /d "%~dp0"

echo ============================================================
echo   个股涨跌概率预测程序 - 本地网页版
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 python，请先安装 Python 3.9+ 并加入 PATH
    pause
    exit /b 1
)

python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [!] 未安装 streamlit，正在安装依赖...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请手动执行： pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo [1/2] 运行自检...
python verify_all.py --skip-heavy
echo.

echo [2/2] 启动网页版（浏览器将自动打开 http://localhost:8501 ）
echo       手机访问：与本机同一 WiFi 时，用 http://本机IP:8501
echo       关闭本窗口即可停止服务。
echo.
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0

pause
