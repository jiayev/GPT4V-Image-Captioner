#!/bin/bash

mod=$1

source ./myenv/bin/activate

python cog_openai_api.py --model $mod
