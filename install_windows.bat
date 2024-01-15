@echo off

set "target_url=www.baidu.com"
set "timeout=3000"

SET PYTHON_VERSION=3.10.2
SET PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe

python --version >NUL 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed. Attempting to install Python %PYTHON_VERSION%...
    powershell -command "Invoke-WebRequest -Uri %PYTHON_INSTALLER_URL% -OutFile python-installer.exe"
    start /wait python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    del /f python-installer.exe
    python --version >NUL 2>&1
    if %ERRORLEVEL% neq 0 (
        echo Failed to install Python.
        exit /b
    )
)

echo Python installed.

ping %target_url% -n 1 -w %timeout% >nul
if %errorlevel% equ 0 (
    echo Use CN
    PowerShell.exe -File "./install_script/install_cn.ps1"
) else (
    echo Use default
    PowerShell.exe -File "./install_script/install.ps1"
)

pause