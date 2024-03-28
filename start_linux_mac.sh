#!/bin/bash
export HF_HOME="huggingface"

source myenv/bin/activate

python ./install_script/check_open.py

python gpt-caption.py "$@"

read -p "Press any key to continue . . . "
