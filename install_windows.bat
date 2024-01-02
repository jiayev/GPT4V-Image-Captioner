@echo off
SETLOCAL EnableExtensions

:: 定义Python版本和安装包下载URL
SET PYTHON_VERSION=3.10.2
SET PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe

:: 检查是否安装了Python
python --version >NUL 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed. Attempting to install Python %PYTHON_VERSION%...
    :: 下载Python安装器
    powershell -command "Invoke-WebRequest -Uri %PYTHON_INSTALLER_URL% -OutFile python-installer.exe"
    
    :: 安装Python，/quiet为静默安装模式
    start /wait python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    
    :: 删除安装器
    del /f python-installer.exe
    
    :: 检查Python是否成功安装
    python --version >NUL 2>&1
    if %ERRORLEVEL% neq 0 (
        echo Failed to install Python.
        exit /b
    )
)

echo Python installed.

:: 创建虚拟环境
python -m venv myenv

:: 激活虚拟环境
call myenv\Scripts\activate

:: 更新pip到最新版本
python -m pip install --upgrade pip

:: 安装必要的Python库
pip install Pillow tqdm gradio requests

:: 运行Python脚本（脚本名称为gpt-caption.py）
python gpt-caption.py
if %ERRORLEVEL% neq 0 (
    echo The script failed to run.
    pause
    exit /b
)

echo Script finished running.
pause
ENDLOCAL