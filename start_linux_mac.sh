#!/bin/bash

source myenv/bin/activate

python gpt-caption.py

read -p "Press any key to continue . . . " -n1 -s
echo