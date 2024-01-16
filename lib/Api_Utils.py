import os
import json
import base64
import requests
import subprocess
import platform
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from huggingface_hub import snapshot_download

API_PATH = 'api_settings.json'

# API使用
def run_openai_api(image_path, prompt, api_key, api_url, quality=None, timeout=10):
    prompt = addition_prompt_process(prompt, image_path)
    # print("prompt{}:",prompt)
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    data = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": f"{quality}"
                    }
                     }
                ]
            }
        ],
        "max_tokens": 300
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 配置重试策略
    retries = Retry(total=5,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"])  # 更新参数名

    with requests.Session() as s:
        s.mount('https://', HTTPAdapter(max_retries=retries))

        try:
            response = s.post(api_url, headers=headers, json=data, timeout=timeout)
            response.raise_for_status()  # 如果请求失败，将抛出 HTTPError
        except requests.exceptions.HTTPError as errh:
            return f"HTTP Error: {errh}"
        except requests.exceptions.ConnectionError as errc:
            return f"Error Connecting: {errc}"
        except requests.exceptions.Timeout as errt:
            return f"Timeout Error: {errt}"
        except requests.exceptions.RequestException as err:
            return f"OOps: Something Else: {err}"

    try:
        response_data = response.json()

        if 'error' in response_data:
            return f"API error: {response_data['error']['message']}"

        caption = response_data["choices"][0]["message"]["content"]
        return caption
    except Exception as e:
        return f"Failed to parse the API response: {e}\n{response.text}"


# 扩展prompt {} 标记功能，从文件读取额外内容
def addition_prompt_process(prompt, image_path):
    # 从image_path分离文件名和扩展名，并更改扩展名为.txt
    if '{' not in prompt and '}' not in prompt:
        return prompt
    file_root, _ = os.path.splitext(image_path)
    new_file_name = os.path.basename(file_root) + ".txt"
    # 从prompt中提取目录路径
    directory_path = prompt[prompt.find('{') + 1: prompt.find('}')]
    # 拼接新的文件路径
    full_path = os.path.join(directory_path, new_file_name)
    # 读取full_path指定的文件内容
    try:
        with open(full_path, 'r') as file:
            file_content = file.read()
    except Exception as e:
        return f"Error reading file: {e}"

    new_prompt = prompt.replace('{' + directory_path + '}', file_content)
    return new_prompt


# API存档
def save_api_details(api_key, api_url):
    settings = {
        'model' : 'GPT',
        'api_key': api_key,
        'api_url': api_url
    }
    # 不记录空的apikey
    if api_key != "":
        with open(API_PATH, 'w', encoding='utf-8') as f:
            json.dump(settings, f)

def save_state(mod):
    settings = {
        'model' : f'Cog-{mod}',
        'api_key': "",
        'api_url': "http://127.0.0.1:8000/v1/chat/completions"
    }
    with open(API_PATH, 'w', encoding='utf-8') as f:
        json.dump(settings, f)
    return f"Set {mod} as default. / {mod}已设为默认"

def get_api_details():
    # 读取API设置
    settings_file = API_PATH
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        return settings.get('model', ''), settings.get('api_key', ''), settings.get('api_url', '')
    return 'GPT', '', ''


# Cog相关
def downloader(model_type, acceleration):
    if acceleration == 'CN':
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    if model_type == 'vqa':
        snapshot_download(
            repo_id="THUDM/cogagent-vqa-hf",
            local_dir="./models/cogagent-vqa-hf",
            max_workers=8
        )
    else:
        snapshot_download(
            repo_id="THUDM/cogagent-chat-hf",
            local_dir="./models/cogagent-chat-hf",
            max_workers=8
        )
    return f"{model_type} Model download completed. / {model_type}模型下载完成"
        
def installer(acceleration):
    if acceleration == 'CN':
        script_path = './install_script/installcog-cn'
    else:
        script_path = './install_script/installcog'

    if platform.system() == "Windows":
        install_command = f'powershell -File "{script_path}.ps1"'
    else:
        install_command = f'bash "{script_path}.sh"'
    try:
        subprocess.run(install_command, check=True, shell=True)
        return f"Env Completed. / 环境配置完成"
    except subprocess.CalledProcessError as e:
        print(f'Error: {e}')