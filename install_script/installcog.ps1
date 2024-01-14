$Env:HF_HOME = "huggingface"

.\myenv\Scripts\activate

Write-Output "Installing deps..."
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -U -I --no-deps xformers==0.0.22
pip install bitsandbytes==0.41.1
pip install ./install_script/deepspeed-0.11.2+8ce7471-py3-none-any.whl
pip install -r ./install_script/require.txt

Write-Output "Install completed"
Read-Host | Out-Null ;
