#!/bin/bash

# Activate the virtual environment
source myenv/bin/activate

# Run Python script
python gpt-caption.py

# Wait for a user input to pause the script
read -p "Press any key to continue . . . " -n1 -s
echo