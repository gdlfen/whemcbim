@echo off
chcp 65001 >nul
title 个股涨跌概率预测 - 命令行版
cd /d "%~dp0"

set SYMBOL=%1
if "%SYMBOL%"=="" set SYMBOL=600519

echo ============================================================
echo   个股涨跌概率预测 - 命令行版
echo   股票代码：%SYMBOL%
echo ============================================================
echo.

python verify_all.py --skip-heavy
if errorlevel 1 (
    echo.
    echo [警告] 自检未全部通过，仍继续分析...
)
echo.

python run_local.py %SYMBOL% --backtest --charts --report
echo.
echo 报告已生成在 reports\ 目录，图表在 charts\ 目录。
echo 用法： 拖入本文件并附加股票代码，或命令行执行  运行分析.bat 600519
pause
