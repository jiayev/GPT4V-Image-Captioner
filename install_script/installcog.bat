@echo off

call myenv\Scripts\activate

set HF_HOME=huggingface
REM 通过百度检测网络设置使用镜像
set "target_url=www.baidu.com"
set "timeout=4000"

ping %target_url% -n 1 -w %timeout% >nul
if %errorlevel% equ 0 (
    echo Use CN
    echo 安装依赖
    set PIP_DISABLE_PIP_VERSION_CHECK=1
    set PIP_NO_CACHE_DIR=1
    set PIP_INDEX_URL=https://mirror.baidu.com/pypi/simple

    echo 安装 torch...
    pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirror.baidu.com/pypi/simple
    if %ERRORLEVEL% neq 0 (
        echo torch 安装失败 > install_temp.txt
        pause >nul
        exit /b 1
    )

) else (
    echo Use default
    echo Installing deps...
    pip install https://download.pytorch.org/whl/cu121/torch-2.2.0%2Bcu121-cp310-cp310-win_amd64.whl
)


pip install ./install_script/deepspeed-0.11.2+8ce7471-py3-none-any.whl
pip install -U -I --no-deps xformers==0.0.24
pip install -r ./install_script/require.txt
if %ERRORLEVEL% neq 0 (
    echo Deps install failed / 依赖安装失败 > install_temp.txt
    pause >nul
    exit /b 1
)

echo Install completed / 安装完毕 > install_temp.txt

pause
