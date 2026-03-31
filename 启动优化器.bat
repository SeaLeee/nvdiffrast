@echo off
chcp 65001 >nul 2>&1
title NvDiffRast Messiah Optimizer

:: ======================================
::  一键启动 NvDiffRast Messiah Optimizer
:: ======================================

cd /d "%~dp0\messiah_optimizer"

echo ============================================
echo   NvDiffRast Messiah Optimizer
echo ============================================
echo.

:: ---- 检查 Python ----
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 未找到 Python，请安装 Python 3.10+ 并加入 PATH。
    goto :fail
)
for /f "tokens=*" %%i in ('python -c "import sys; print(sys.version.split()[0])"') do set PYVER=%%i
echo [OK] Python %PYVER%

:: ---- 检查 PyTorch + CUDA ----
python -c "import torch; assert torch.cuda.is_available(), 'no cuda'" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] PyTorch CUDA 不可用，尝试安装...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    if %errorlevel% neq 0 (
        echo [ERROR] PyTorch 安装失败。
        goto :fail
    )
)
for /f "tokens=*" %%i in ('python -c "import torch; print(torch.__version__, '- CUDA', torch.version.cuda, '- GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"') do echo [OK] PyTorch %%i

:: ---- 检查 PyQt6 ----
python -c "import PyQt6" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] 安装依赖...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] 依赖安装失败。
        goto :fail
    )
)
echo [OK] PyQt6

:: ---- 检查 nvdiffrast ----
python -c "import nvdiffrast.torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] nvdiffrast 未安装，尝试编译...
    call "%~dp0\build_nvdiffrast.bat"
    python -c "import nvdiffrast.torch" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [WARN] nvdiffrast 编译失败，将使用软件渲染。
    ) else (
        echo [OK] nvdiffrast 编译成功
    )
) else (
    echo [OK] nvdiffrast
)

:: ---- 启动 ----
echo.
echo [INFO] 启动编辑器...
echo ============================================
echo.
python editor\main.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 程序异常退出 (code %errorlevel%)
    goto :fail
)
goto :end

:fail
echo.
pause

:end
