import base64
import requests
import json
import os
import gradio as gr
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import mimetypes
import shutil
import threading
import subprocess

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

def modify_file_content(file_path, new_content, mode):

    if mode == "skip/跳过" and os.path.exists(file_path):
        print(f"Skip writing, as the file {file_path} already exists.")
        return

    if mode == "overwrite/覆盖" or not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        return

    with open(file_path, 'r+', encoding='utf-8') as file:
        existing_content = file.read()
        file.seek(0)
        if mode == "prepend/前置插入":
            combined_content = unique_elements(new_content, existing_content)
            file.write(combined_content)
            file.truncate()
        elif mode == "append/结尾追加":
            combined_content = unique_elements(existing_content, new_content)
            file.write(combined_content)
            file.truncate()
        else:
            raise ValueError("Invalid mode. Must be 'overwrite/覆盖', 'prepend/前置插入', or 'append/结尾追加'.")

should_stop = threading.Event()

def get_saved_api_details():
    settings_file = 'api_settings.json'
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        return settings.get('api_key', ''), settings.get('api_url', '')
    return '', ''

def save_api_details(api_key, api_url):
    settings = {
        'api_key': api_key,
        'api_url': api_url
    }
    with open('api_settings.json', 'w', encoding='utf-8') as f:
        json.dump(settings, f)

def run_openai_api(image_path, prompt, api_key, api_url):
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    
    data = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
                ]
            }
        ],
        "max_tokens": 300
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.post(api_url, headers=headers, json=data, timeout=10)
    
    if response.status_code != 200:
        return f"Error: API request failed with status {response.status_code}\n{response.text}"

    try:
        response_data = response.json()
        
        if 'error' in response_data:
            return f"API error: {response_data['error']['message']}"
        
        caption = response_data["choices"][0]["message"]["content"]
        return caption
    except Exception as e:
        return f"Failed to parse the API response: {e}\n{response.text}"

def process_single_image(api_key, prompt, api_url, image_path):
    save_api_details(api_key, api_url)
    caption = run_openai_api(image_path, prompt, api_key, api_url)
    print(caption)
    return caption

def process_batch_images(api_key, prompt, api_url, image_dir, file_handling_mode):
    should_stop.clear()
    save_api_details(api_key, api_url)
    results = []

    supported_image_formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif')
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(supported_image_formats):
                image_files.append(os.path.join(root, file))

    def process_image(filename, file_handling_mode):
        image_path = os.path.join(image_dir, filename)
        base_filename = os.path.splitext(filename)[0]
        caption_filename = f"{base_filename}.txt"
        caption_path = os.path.join(image_dir, caption_filename)

        if file_handling_mode != "skip/跳过" or not os.path.exists(caption_path):
            caption = run_openai_api(image_path, prompt, api_key, api_url)
            
            if caption.startswith("Error:") or caption.startswith("API error:"):
                return handle_error(image_path, caption_path, caption_filename, filename)
            else:
                modify_file_content(caption_path, caption, file_handling_mode)
                return filename, caption_path
        else:
            return filename, "Skipped because caption file already exists."

    def handle_error(image_path, caption_path, caption_filename, filename):
        parent_dir = os.path.dirname(image_dir)
        error_image_dir = os.path.join(parent_dir, "error_images")
        if not os.path.exists(error_image_dir):
            os.makedirs(error_image_dir)

        error_image_path = os.path.join(error_image_dir, filename)
        error_caption_path = os.path.join(error_image_dir, caption_filename)
        
        try:
            shutil.move(image_path, error_image_path)
            if os.path.exists(caption_path):
                shutil.move(caption_path, error_caption_path)
            return filename, "Error handled and image with its caption moved to error directory."
        except Exception as e:
            return filename, f"An unexpected error occurred while moving {filename} or {caption_filename}: {e}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_image, filename, file_handling_mode) for filename in image_files]
        progress = tqdm(total=len(futures), desc="Processing images")
        try:
            for future in as_completed(futures):
                if should_stop.is_set():
                    for f in futures:
                        f.cancel()
                    print("Batch processing was stopped by the user.")
                    break
                try:
                    result = future.result()
                except Exception as e:
                    result = None
                    print(f"An exception occurred: {e}")
                results.append(result)
                progress.update(1)
        finally:
            progress.close()
            executor.shutdown(wait=False)

    return results

saved_api_key, saved_api_url = get_saved_api_details()

def run_script(folder_path, keywords):
    keywords = keywords if keywords else "sorry,error"
    result = subprocess.run(
        [
            'python', 'Failed_Tagging_File_Screening.py',
            '--image_path', folder_path,
            '--keywords', keywords
        ],
        capture_output=True, text=True
    )
    return result.stdout if result.stdout else "No Output", result.stderr if result.stderr else "No Error"

def stop_batch_processing():
    should_stop.set()
    return "Attempting to stop batch processing. Please wait for the current image to finish."

with gr.Blocks(title="GPT4V captioner") as demo:
    gr.Markdown("### Image Captioning with GPT-4-Vision API / 使用 GPT-4-Vision API 进行图像打标")
    
    with gr.Row():
        api_key_input = gr.Textbox(label="API Key", placeholder="Enter your GPT-4-Vision API Key here", type="password", value=saved_api_key)
        api_url_input = gr.Textbox(label="API URL", value=saved_api_url or "https://api.openai.com/v1/chat/completions", placeholder="Enter the GPT-4-Vision API URL here")
    prompt_input = gr.Textbox(label="Prompt",
                              value="As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. Prioritize the tags by relevance. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and camera specifications, and any other tags crucial for the image. When tagging photos of people, include specific details like gender, nationality, attire, actions, pose, expressions, accessories, makeup, composition type, age, etc. For other image categories, apply appropriate and common descriptive tags as well. Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. Your tags should be accurate, non-duplicative, and within a 20-75 word count range. These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. Tags should be comma-separated. Exceptional tagging will be rewarded with $10 per image.",
                              placeholder="Enter a descriptive prompt",
                              lines=5)
    with gr.Tab("Single Image Processing / 单图处理"):
        with gr.Row():
            image_input = gr.Image(type='filepath', label="Upload Image / 上传图片")
            single_image_output = gr.Textbox(label="Caption Output / 标签输出")
        with gr.Row():
            single_image_submit = gr.Button("Caption Single Image / 图片打标", variant='primary')
        
    
    with gr.Tab("Batch Image Processing / 批量图像处理"):
        with gr.Row():
            batch_dir_input = gr.Textbox(label="Batch Directory / 批量目录", placeholder="Enter the directory path containing images for batch processing")
        with gr.Row():
            batch_process_submit = gr.Button("Batch Process Images / 批量处理图像", variant='primary')        
        with gr.Row():
            batch_output = gr.Textbox(label="Batch Processing Output / 批量输出")
            file_handling_mode = gr.Radio(
            choices=["overwrite/覆盖", "prepend/前置插入", "append/结尾追加", "skip/跳过"],  
            value="overwrite/覆盖", 
            label="If a caption file exists: / 如果已经存在打标文件: "
        )
        with gr.Row():
            stop_button = gr.Button("Stop Batch Processing / 停止批量处理")
            stop_button.click(stop_batch_processing, inputs=[], outputs=batch_output)
    
    with gr.Tab("Failed Tagging File Screening / 打标失败文件筛查"):
        folder_input = gr.Textbox(label="Folder Input / 文件夹输入", placeholder="Enter the directory path")
        keywords_input = gr.Textbox(placeholder="Enter keywords, e.g., sorry,error / 请输入关键词，例如：sorry,error", label="Keywords (optional) / 关键词（可选）")
        run_button = gr.Button("Run Script / 运行脚本", variant='primary')
        output_area = gr.Textbox(label="Script Output / 脚本输出")
        
        run_button.click(fn=run_script, inputs=[folder_input, keywords_input], outputs=output_area)

             
    def batch_process(api_key, api_url, prompt, batch_dir, file_handling_mode):
        process_batch_images(api_key, prompt, api_url, batch_dir, file_handling_mode) 
        return "Batch processing complete. Captions saved or updated as '.txt' files next to images."
    
    def caption_image(api_key, api_url, prompt, image):
        if image:
            return process_single_image(api_key, prompt, api_url, image)

    single_image_submit.click(caption_image, inputs=[api_key_input, api_url_input, prompt_input, image_input], outputs=single_image_output)
    batch_process_submit.click(
        batch_process, 
        inputs=[api_key_input, api_url_input, prompt_input, batch_dir_input, file_handling_mode], 
        outputs=batch_output
    )

    gr.Markdown("### Developers: Jiaye, LEOSAM")
    gr.Markdown("##### 开发人员：Jiaye、LEOSAM 是只兔狲")

if __name__ == "__main__":
    demo.launch()
