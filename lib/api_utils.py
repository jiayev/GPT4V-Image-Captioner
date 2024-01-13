# api_utils.py
import json
import os

def save_api_details(api_key, api_url):
    settings = {
        'api_key': api_key,
        'api_url': api_url
    }
    # 不记录空的apikey
    if api_key != "":
        with open('api_settings.json', 'w', encoding='utf-8') as f:
            json.dump(settings, f)


def get_api_details():
    # 读取API设置
    settings_file = 'api_settings.json'
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        return settings.get('api_key', ''), settings.get('api_url', '')
    return '', ''
