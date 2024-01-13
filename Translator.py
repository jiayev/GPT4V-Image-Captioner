import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor, as_completed

class ChineseTranslator:
    def __init__(self):
        self.client = requests.Session()

    def translate(self, text):
        if not text:
            return None

        to_lang = to_lang.replace("-CN", "").replace("-TW", "")
        payload = {
            "appid": "105",
            "sgid": "en",
            "sbid": "en",
            "egid": "zh-CN",
            "ebid": "zh-CN",
            "content": text,
            "type": "2",
        }

        response = self.client.post("https://translate-api-fykz.xiangtatech.com/translation/webs/index", data=payload)
        if response.status_code == 200:
            json_data = response.json()
            by_value = json_data.get("by", "")
            if not by_value:
                return None
            return by_value

        return None

    def close_session(self):
        self.client.close()
        
class GPTTranslator:
    def __init__(self, api_key, api_url):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

        self.api_url = api_url

    def translate(self, text):
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": f"你是一个英译中专家，请直接返回'{text}'最有可能的两种中文翻译结果，结果以逗号间隔."}
            ]
        }
        response = self.session.post(self.api_url, headers=self.headers, json=data)
        response_data = response.json()

        if response.status_code == 200 and 'choices' in response_data and 'content' in response_data['choices'][0]['message']:
            return response_data['choices'][0]['message']['content']
        else:
            return f"Error or no translation for tag: {text}"

    def close_session(self):
        self.session.close()
        
def translate_tags(translator, tags):
    translations = [None] * len(tags)
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_index = {executor.submit(translator.translate, tag): i for i, tag in enumerate(tags)}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            translations[index] = future.result()
            
    translator.close_session()
            
    return translations
