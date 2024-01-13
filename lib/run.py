import os
import base64
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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


def unique_elements(original, addition):
    original_list = list(map(str.strip, original.split(',')))
    addition_list = list(map(str.strip, addition.split(',')))

    combined_list = []
    seen = set()
    for item in original_list + addition_list:
        if item not in seen and item != '':
            seen.add(item)
            combined_list.append(item)

    return ', '.join(combined_list)