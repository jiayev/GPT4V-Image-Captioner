@echo off
set url=http://127.0.0.1:8848
set max_attempts=40

call myenv\Scripts\activate
start /B python gpt-caption.py
:LOOP
ping -n 1 %url% >nul
if errorlevel 1 (
    echo Waiting for %url% to respond...
    timeout /nobreak /t 2 >nul
    set /a max_attempts -=1
    if %max_attempts% gtr 0 goto :LOOP
    echo Maximum attempts reached. Exiting.
    exit /b 1
) else (
    echo %url% is reachable. Opening...
    start "" %url%
    exit /b 0
)
pause
