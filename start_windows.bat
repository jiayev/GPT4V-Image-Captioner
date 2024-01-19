@echo off
set url=http://127.0.0.1:8848
set HF_HOME=huggingface

call myenv\Scripts\activate
start /B python gpt-caption.py
timeout /nobreak /t 2 >nul
start "" %url%
pause
