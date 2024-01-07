$Env:HF_HOME = "huggingface"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
$Env:PIP_INDEX_URL = "https://mirror.baidu.com/pypi/simple"

function InstallFail {
    Write-Output "安装失败。"
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
Check "激活虚拟环境失败。"

pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirror.baidu.com/pypi/simple
Check "torch 安装失败"
pip install -U -I --no-deps xformers==0.0.22
Check "xformers 安装失败。"

Write-Output "安装 bitsandbytes..."
pip install bitsandbytes==0.41.1 --index-url https://jihulab.com/api/v4/projects/140618/packages/pypi/simple
Write-Output "安装 deepspeed..."
pip install deepspeed-0.11.2+8ce7471-py3-none-any.whl

Write-Output "安装依赖..."
pip install -r require.txt
Check "依赖安装失败。"

Write-Output "安装完毕"
Read-Host | Out-Null ;