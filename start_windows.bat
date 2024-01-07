@echo off
call myenv\Scripts\activate
start /B python gpt-caption.py
start "" "http://127.0.0.1:8848"
pause
