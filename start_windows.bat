@echo off
set url=http://127.0.0.1:8848
set HF_HOME=huggingface

call myenv\Scripts\activate
python ./install_script/check_open.py

python gpt-caption.py

pause
