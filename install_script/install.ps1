$Env:HF_HOME = "huggingface"

if (!(Test-Path -Path "myenv")) {
    python -m venv myenv
}

.\myenv\Scripts\activate

python -m pip install --upgrade pip

pip install -r ./install_script/requirements.txt

Write-Output "Install completed"
Read-Host | Out-Null ;