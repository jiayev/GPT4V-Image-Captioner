export HF_HOME="huggingface"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export PIP_INDEX_URL="https://mirror.baidu.com/pypi/simple"

function install_fail {
    echo "��װʧ�ܡ�"
    read -p "Press enter to continue"
    exit 1
}

function check {
    if [ $? -ne 0 ]; then
        echo $1
        install_fail
    fi
}

source ./myenv/bin/activate
check "�������⻷��ʧ�ܡ�"

pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirror.baidu.com/pypi/simple
Check "torch ��װʧ��"

echo "��װ bitsandbytes..."
pip install bitsandbytes==0.41.1 --index-url https://jihulab.com/api/v4/projects/140618/packages/pypi/simple

echo "��װ deepspeed..."
pip install deepspeed

echo "��װ����..."
pip install -r ./install_script/require.txt
check "������װʧ�ܡ�"

echo "��װ���"
read -p "Press enter to continue"