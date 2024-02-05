@echo off
set HF_HOME=huggingface

call myenv\Scripts\activate
python ./install_script/check_open.py

python gpt-caption.py

pause
