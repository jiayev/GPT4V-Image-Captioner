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

if (!(Test-Path -Path "myenv")) {
    Write-Output "正在创建虚拟环境..."
    python -m venv myenv
    Check "创建虚拟环境失败，请检查 python 是否安装完毕以及 python 版本是否为64位版本的python 3.10、或python的目录是否在环境变量PATH内。"
}

.\myenv\Scripts\activate
Check "激活虚拟环境失败。"

python -m pip install --upgrade pip

Write-Output "安装依赖..."
pip install -r requirements.txt
Check "依赖安装失败。"

Write-Output "安装完毕，请运行Start打开GUI"
Read-Host | Out-Null ;