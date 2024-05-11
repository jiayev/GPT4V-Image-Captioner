import io
import os
import time
import json
import base64
import requests
import subprocess
import platform
from PIL import Image
from requests.adapters import HTTPAdapter
import re
from urllib3.util.retry import Retry
from huggingface_hub import snapshot_download

API_PATH = 'api_settings.json'
QWEN_MOD = 'qwen-vl-plus'
DEFAULT_GPT_MODEL = 'gpt-4-vision-preview'
DEFAULT_CLAUDE_MODEL = 'claude-3-sonnet'

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

# 通义千问VL
def is_ali(api_url):
    if api_url.endswith("/v1/services/aigc/multimodal-generation/generation"):
        return True
    else:
        return False
    
def is_claude(api_url, model):
    if api_url.endswith("v1/messages") or "claude" in model.lower():
        return True
    else:
        return False

def qwen_api_switch(mod):
    global QWEN_MOD
    QWEN_MOD = mod
    return QWEN_MOD

def qwen_api(image_path, prompt, api_key):
    print(f"QWEN_MOD: {QWEN_MOD}")

    os.environ['DASHSCOPE_API_KEY'] = api_key
    from dashscope import MultiModalConversation
    img = f"file://{image_path}"
    messages = [{
        'role': 'system',
        'content': [
            {'text': 'You are a helpful assistant.'}
            ]
        }, {
        'role':'user',
        'content': [
            {'image': img},
            {'text': prompt},
            ]
        }]

    response = MultiModalConversation.call(model=QWEN_MOD, messages=messages, stream=False, max_length=300)
    if '"status_code": 400' in response:
        return f"API error: {response}"
    if response.get("output") and response["output"].get("choices") and response["output"]["choices"][0].get("message") and response["output"]["choices"][0]["message"].get("content"):
        if response["output"]["choices"][0]["message"]["content"][0].get("text", False):
            caption = response["output"]["choices"][0]["message"]["content"][0]["text"]
        else:
            box_value = response["output"]["choices"][0]["message"]["content"][0]["box"]
            text_value = response["output"]["choices"][0]["message"]["content"][1]["text"]
            b_value = re.search(r'<ref>(.*?)</ref>', box_value).group(1)
            caption = b_value + text_value
    else:
        caption = response
    return caption

def claude_api(image_path, prompt, api_key, api_url, model, quality=None):
    print(f"CLAUDE_MODEL: {model}")
    
    with open(image_path, "rb") as image_file:
        # Downscale the image
        image = Image.open(image_file)
        width, height = image.size
        if quality:
            if quality == "high":
                target = 1024
            elif quality == "low":
                target = 512
            elif quality == "auto":
                if width >= 1024 or height >= 1024:
                    target = 1024
                else:
                    target = 512
        else:
            target = 1024
            
        aspect_ratio = width / height

        # Determine the new dimensions while maintaining the aspect ratio
        if width > target or height > target:
            if width > height:
                new_width = target
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = target
                new_width = int(new_height * aspect_ratio)
        else:
            new_width, new_height = width, height

        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # Use buffer to store image
        buffer = io.BytesIO()
        resized_image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Claude API
    data = {
        "model": model,
        "max_tokens": 300,
        "system": prompt,
        "messages": [
            {"role": "user", "content": [
                    {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    },
                    {"type": "text", "text": prompt}
                ]  
            }
        ]
    }

    # print(f"data: {data}\n")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "anthropic-version": "2023-06-01"
    }

    # 配置重试策略
    retries = Retry(total=5,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"])  # 更新参数名
    
    with requests.Session() as s:
        s.mount('https://', HTTPAdapter(max_retries=retries))

        try:
            response = s.post(api_url, headers=headers, json=data)
            response.raise_for_status()
        # 连接错误回显
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
        caption = response_data['content'][0]['text']
        return caption
    except Exception as e:
        return f"Failed to parse the API response: {e}\n{response.text}"
    


# API使用
def run_openai_api(image_path, prompt, api_key, api_url, quality=None, timeout=10, model=DEFAULT_GPT_MODEL):
    prompt = addition_prompt_process(prompt, image_path)
    # print("prompt{}:",prompt)

    # Qwen-VL
    if is_ali(api_url):
        return qwen_api(image_path, prompt, api_key)
    if is_claude(api_url, model):
        return claude_api(image_path, prompt, api_key, api_url, model, quality)
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    # GPT-4V
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content":
                [
                    {"type": "image_url", "image_url":
                        {"url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": f"{quality}"}
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "max_tokens": 300
    }

    print(f"data: {data}\n")

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
            response.raise_for_status()
        # 连接错误回显
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


# API存档
def save_api_details(api_key, api_url):
    if is_ali(api_url):
        settings = {
        'model' : QWEN_MOD,
        'api_key': api_key,
        'api_url': api_url
        }
    else:
        settings = {
            'model' : 'GPT',
            'api_key': api_key,
            'api_url': api_url
        }
    # 不记录空的apikey
    if api_key != "":
        with open(API_PATH, 'w', encoding='utf-8') as f:
            json.dump(settings, f)

def save_state(llm, key, url):
    if llm[:3] == "GPT" or llm[:4] == "qwen":
        settings = {
            'model': llm,
            'api_key': key,
            'api_url': url
        }

    elif llm[:3] == "Cog" or llm[:4] == "moon" or llm[:4] == "omni":
        settings = {
            'model' : llm,
            'api_key': "",
            'api_url': "http://127.0.0.1:8000/v1/chat/completions"
        }

    output = f"Set {llm} as default. / {llm}已设为默认"
    with open(API_PATH, 'w', encoding='utf-8') as f:
        json.dump(settings, f)
    return output

# 读取API设置
def get_api_details():
    settings_file = API_PATH
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        if settings.get('model', '') != '':
            mod = settings.get('model', '')
            url = settings.get('api_url', '')
            if mod[:4] == "qwen":
                global QWEN_MOD
                QWEN_MOD = mod
            else:
                if is_ali(url):
                    mod = QWEN_MOD
            return mod, settings.get('api_key', ''), url
        else:
            if settings.get('api_key', '') != '':
                i_key = settings.get('api_key', '')
                i_url = settings.get('api_url', '')
                save_api_details(i_key,i_url)
                with open(settings_file, 'r') as i:
                    settings = json.load(i)
                return settings.get('model', ''), settings.get('api_key', ''), settings.get('api_url', '')
    return 'GPT', '', ''


# 本地模型相关
def downloader(model_type, acceleration):
    endpoint = 'https://hf-mirror.com' if acceleration == 'CN' else None
    if model_type == 'vqa' or model_type == 'chat':
        snapshot_download(
            repo_id="lmsys/vicuna-7b-v1.5",
            allow_patterns=["tokenizer*","special_tokens_map.json"],
            endpoint=endpoint
        )
    if model_type == 'vqa':
        snapshot_download(
            repo_id="THUDM/cogagent-vqa-hf",
            local_dir="./models/cogagent-vqa-hf",
            max_workers=8,
            endpoint=endpoint
        )
    elif model_type == 'chat':
        snapshot_download(
            repo_id="THUDM/cogagent-chat-hf",
            local_dir="./models/cogagent-chat-hf",
            max_workers=8,
            endpoint=endpoint
        )
    elif model_type == 'moon':
        snapshot_download(
            repo_id="vikhyatk/moondream1",
            local_dir="./models/moondream",
            max_workers=8,
            endpoint=endpoint
        )
    elif model_type == 'omni':
        snapshot_download(
            repo_id="openbmb/OmniLMM-12B",
            local_dir="./models/OmniLMM-12B",
            max_workers=8,
            endpoint=endpoint
        )
    return f"{model_type} Model download completed. / {model_type}模型下载完成"

def installer():
    if platform.system() == "Windows":
        install_command = f'.\install_script\installcog.bat'
    else:
        install_command = f'./install_script/installcog.sh'
        subprocess.Popen(f'chmod +x {install_command}', shell=True)
        subprocess.Popen('', shell=True) #Use an empty subprocess to refresh permission. If deleted, installcog.sh wouldn't launch properly, with Permission denied error
    subprocess.Popen(install_command, shell=True)

    while not os.path.exists('install_temp.txt'):
        time.sleep(2)
    with open('install_temp.txt', 'r') as file:
        result_string = file.read()
    os.remove('install_temp.txt')
    return result_string
