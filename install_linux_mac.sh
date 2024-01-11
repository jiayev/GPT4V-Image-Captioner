#!/bin/bash

# Define the Python version
PYTHON_VERSION=3.10.2

# Check if Python is installed and the version is as expected
if ! command -v python3 --version &>/dev/null || ! python3 --version | grep -q "$PYTHON_VERSION"; then
    echo "Python is not installed or not the expected version. Please install Python $PYTHON_VERSION."
    exit 1
fi

echo "Python is installed."

# Create a virtual environment
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# Upgrade pip to the latest version
pip install --upgrade pip

# Install necessary Python libraries
pip install -r ./install_script/requirements.txt

# Run Python script (script name is gpt-caption.py)
python3 gpt-caption.py
if [ $? -ne 0 ]; then
    echo "The script failed to run."
    exit 1
fi

echo "Script finished running."
