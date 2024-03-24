#!/bin/bash

source myenv/bin/activate

export HF_HOME="huggingface"

target_url="www.baidu.com"
timeout=4000
ping -c 1 -W $timeout $target_url > /dev/null

if [ $? -eq 0 ]; then
    echo "Use CN"
    echo "安装依赖"

    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PIP_NO_CACHE_DIR=1
    export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

    echo "安装 torch..."
    pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple
    if [ $? -ne 0 ]; then
        echo "torch 安装失败" > install_temp.txt
        exit 1
    fi

else
    echo "Use default"
    echo "Installing deps..."
    pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
fi

pip install deepspeed
pip install -U -I --no-deps xformers==0.0.25
pip install -r ./install_script/require.txt
if [ $? -ne 0 ]; then
    echo "Deps install failed / 依赖安装失败" > install_temp.txt
    exit 1
fi

echo "Install completed / 安装完毕" > install_temp.txt

read -p "Press Enter to continue..."

