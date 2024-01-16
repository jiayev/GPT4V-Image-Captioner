export HF_HOME="huggingface"

source myenv/bin/activate

echo "Installing deps..."
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes==0.41.1
pip install deepspeed
pip install -r ./install_script/require.txt

echo "Install completed"
read -p "Press enter to continue"
