@echo off
REM Minimal PATH to avoid cmd line length limit
set PATH=C:\Program Files\Python310;C:\Program Files\Python310\Scripts;C:\Users\lixiaoxi1\AppData\Roaming\Python\Python310\Scripts;C:\Windows\System32;C:\Windows
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin;%PATH%
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.9
cd /d d:\github\nvdiffrastsea
if exist build rmdir /s /q build
echo === Starting build ===
python setup.py develop --user
echo === Build exit code: %ERRORLEVEL% ===
