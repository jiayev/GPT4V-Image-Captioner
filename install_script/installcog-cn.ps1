$Env:HF_HOME = "huggingface"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
$Env:PIP_INDEX_URL = "https://mirror.baidu.com/pypi/simple"

function InstallFail {
    Write-Output "��װʧ�ܡ�"
    Read-Host | Out-Null ;
    Exit
}

function Check {
    param (
        $ErrorInfo
    )
    if (!($?)) {
        Write-Output $ErrorInfo
        InstallFail
    }
}

.\myenv\Scripts\activate
Check "�������⻷��ʧ�ܡ�"

pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirror.baidu.com/pypi/simple
Check "torch ��װʧ��"

Write-Output "��װ bitsandbytes..."
pip install bitsandbytes==0.41.1 --index-url https://jihulab.com/api/v4/projects/140618/packages/pypi/simple

Write-Output "��װ deepspeed..."
pip install ./install_script/deepspeed-0.11.2+8ce7471-py3-none-any.whl

Write-Output "��װ����..."
pip install -r ./install_script/require.txt
Check "������װʧ�ܡ�"

Write-Output "��װ���"
Read-Host | Out-Null ;