#!/bin/bash

# Define the Python version
PYTHON_VERSION=3.10.

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

# Ping google to decide if use mirror
target_url="www.google.com"
timeout=3000
ping -c 1 -W $timeout $target_url -w 3 > /dev/null

if [ $? -ne 0 ]; then
    echo "Use CN"
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PIP_NO_CACHE_DIR=1
    export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo "Use default"
fi

# Upgrade pip to the latest version
pip install --upgrade pip

# Install necessary Python libraries
pip install -r ./install_script/requirements.txt

echo ""
echo "Install completed, please run Start to open the GUI"
echo "安装完毕，请运行Start打开GUI"
echo ""
read -p "press any key to continue...
按任意键继续..."
