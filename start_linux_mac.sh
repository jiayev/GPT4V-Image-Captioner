#!/bin/bash
url="http://127.0.0.1:8848"
export HF_HOME="huggingface"

source myenv/bin/activate

python gpt-caption.py &
sleep 2
xdg-open "$url"

read -p "Press any key to continue . . . " -n1 -s
echo