@echo off

REM 检测python安装
SET PYTHON_VERSION=3.10.2
SET PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe

python --version >NUL 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed. Attempting to install Python %PYTHON_VERSION%...
    bitsadmin /transfer "PythonInstaller" %PYTHON_INSTALLER_URL% python-installer.exe
    start /wait python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    del /f python-installer.exe
    python --version >NUL 2>&1
    if %ERRORLEVEL% neq 0 (
        echo Failed to install Python.
        pause >nul
        exit /b 1
    )
)

echo Python installed.


REM 虚拟环境检测创建
if not exist "myenv" (
    echo 正在创建虚拟环境...
    python -m venv myenv
    if %ERRORLEVEL% neq 0 (
        echo 创建虚拟环境失败，请检查 python 是否安装完毕以及 python 版本是否为64位版本的python 3.10、或python的目录是否在环境变量PATH内。
        pause >nul
        exit /b 1
    )
)

call myenv\Scripts\activate


REM 通过谷歌检测网络设置使用镜像
set "target_url=www.google.com"
set "timeout=3000"

ping %target_url% -n 1 -w %timeout% >nul
if %errorlevel% neq 0 (
    echo Use CN
    set PIP_DISABLE_PIP_VERSION_CHECK=1
    set PIP_NO_CACHE_DIR=1
    set PIP_INDEX_URL=https://mirror.baidu.com/pypi/simple
) else (
    echo Use default
)

set HF_HOME=huggingface

REM 安装依赖

echo Installing deps...
echo 安装依赖
python -m pip install --upgrade pip
pip install -r ./install_script/requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Deps install failed
    echo 依赖安装失败。
    pause >nul
    exit /b 1
)

echo.
echo Install completed, please run Start to open the GUI
echo 安装完毕，请运行Start打开GUI
echo.

pause