import gradio as gr
import argparse
import os
import shutil
import threading

import concurrent.futures
from tqdm import tqdm

import subprocess
import time
import requests
import socket

from lib.Img_Processing import process_images_in_folder, run_script
from lib.Tag_Processor import modify_file_content, process_tags
from lib.GPT_Prompt import get_prompts_from_csv, save_prompt, delete_prompt
from lib.Api_Utils import run_openai_api, save_api_details, get_api_details, downloader, installer, save_state, qwen_api_switch
from lib.Detecter import detecter

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
mod_default, saved_api_key, saved_api_url = get_api_details()
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif')

# 图像打标
should_stop = threading.Event()
def stop_batch_processing():
    should_stop.set()
    return "Attempting to stop batch processing. Please wait for the current image to finish."

def process_single_image(api_key, prompt, api_url, image_path, quality, timeout):
    save_api_details(api_key, api_url)
    caption = run_openai_api(image_path, prompt, api_key, api_url, quality, timeout)
    print(caption)
    return caption

def process_batch_images(api_key, prompt, api_url, image_dir, file_handling_mode, quality, timeout):
    should_stop.clear()
    save_api_details(api_key, api_url)
    results = []

    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_IMAGE_FORMATS):
                image_files.append(os.path.join(root, file))

    def process_image(filename, file_handling_mode):
        image_path = os.path.join(image_dir, filename)
        base_filename = os.path.splitext(filename)[0]
        caption_filename = f"{base_filename}.txt"
        caption_path = os.path.join(image_dir, caption_filename)

        if file_handling_mode != "skip/跳过" or not os.path.exists(caption_path):
            caption = run_openai_api(image_path, prompt, api_key, api_url, quality, timeout)

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
        futures = {}
        for filename in image_files:
            future = executor.submit(process_image, filename, file_handling_mode)
            futures[future] = filename  # 将 future 和 filename 映射起来
        progress = tqdm(total=len(futures), desc="Processing images")

        try:
            for future in concurrent.futures.as_completed(futures):
                filename = futures[future]
                if should_stop.is_set():
                    for f in futures:
                        f.cancel()
                    print("Batch processing was stopped by the user.")
                    break
                try:
                    result = future.result()
                except Exception as e:
                    result = (filename, f"An exception occurred: {e}")
                    print(f"An exception occurred while processing {filename}: {e}")
                results.append(result)
                progress.update(1)
        finally:
            progress.close()
            executor.shutdown(wait=False)

    print(f"Processing complete. Total images processed: {len(results)}")
    return results

def handle_file(image_path, target_path, file_handling_mode):
    try:
        if file_handling_mode[:4] == "copy":
            shutil.copy(image_path, target_path)
        elif file_handling_mode[:4] == "move":
            shutil.move(image_path, target_path)
    except Exception as e:
        print(f"An exception occurred while handling the file {image_path}: {e}")
        return f"Error handling file {image_path}: {e}"
    return

def process_batch_watermark_detection(api_key, prompt, api_url, image_dir, detect_file_handling_mode, quality, timeout,
                                      watermark_dir):
    should_stop.clear()
    save_api_details(api_key, api_url)
    results = []
    prompt = 'Is image have watermark'

    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_IMAGE_FORMATS):
                image_files.append(os.path.join(root, file))

    def process_image(filename, detect_file_handling_mode, watermark_dir):
        image_path = os.path.join(image_dir, filename)
        caption = run_openai_api(image_path, prompt, api_key, api_url, quality, timeout)

        if caption.startswith("Error:") or caption.startswith("API error:"):
            return "error"

        # EOI是cog迷之误判？
        if 'Yes,' in caption and '\'EOI\'' not in caption:
            target_path = os.path.join(watermark_dir, filename)
            handle_file(filename, watermark_dir, detect_file_handling_mode)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for filename in image_files:
            future = executor.submit(process_image, filename, detect_file_handling_mode, watermark_dir)
            futures[future] = filename  # 将 future 和 filename 映射起来
        progress = tqdm(total=len(futures), desc="Processing images")

        try:
            for future in concurrent.futures.as_completed(futures):
                filename = futures[future]  # 获取正在处理的文件名
                if should_stop.is_set():
                    for f in futures:
                        f.cancel()
                    print("Batch processing was stopped by the user.")
                    break
                try:
                    result = future.result()
                except Exception as e:
                    result = (filename, f"An exception occurred: {e}")
                    print(f"An exception occurred while processing {filename}: {e}")
                results.append(result)
                progress.update(1)
        finally:
            progress.close()
            executor.shutdown(wait=False)

    results = f"Total checked images: {len(results)}"
    return results

def classify_images(api_key, api_url, quality, prompt, timeout, detect_file_handling_mode, image_dir, o_dir, *list_r):

    # 初始化
    should_stop.clear()
    save_api_details(api_key, api_url)
    results = []

    # 检查输入
    if not os.path.exists(image_dir):
        return "Error: Image directory does not exist. / 错误：图片目录不存在"
    if not o_dir:
        o_dir = os.path.join(image_dir, "classify_output")
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)

    # 获取图像
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_IMAGE_FORMATS):
                image_files.append(os.path.join(root, file))

    # 转换列表
    rules = []
    for i in range(0, len(list_r), 2):
        rule_type = list_r[i]
        rule_input = list_r[i + 1]
        if rule_type and rule_input:
            rule_type_bool = rule_type == "Involve / 包含"
            rules.append((rule_type_bool, rule_input))
    if rules == []:
        return "Error: All rules are empty. / 错误：未设置规则"

    # 图像处理
    def process_image(filename, rules, detect_file_handling_mode, image_dir, o_dir):
        image_path = os.path.join(image_dir, filename)
        caption = run_openai_api(image_path, prompt, api_key, api_url, quality, timeout)

        if caption.startswith("Error:") or caption.startswith("API error:"):
            return "error"

        matching_rules = []
        for rule_bool, rule_input in rules:
            if (rule_bool and rule_input in caption) or (not rule_bool and rule_input not in caption):
                matching_rules.append(rule_input)

        if matching_rules:
            folder_name = "-".join(matching_rules)
            target_folder = os.path.join(o_dir, folder_name)
            os.makedirs(target_folder, exist_ok=True)
            handle_file(filename, target_folder, detect_file_handling_mode)
        elif matching_rules == []:
            no_match_folder = os.path.join(o_dir, "no_match")
            os.makedirs(no_match_folder, exist_ok=True)
            handle_file(filename, no_match_folder, detect_file_handling_mode)

    # 批量处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {}
        for filename in image_files:
            future = executor.submit(process_image, filename, rules, detect_file_handling_mode, image_dir, o_dir)
            futures[future] = filename  # 将 future 和 filename 映射起来
        progress = tqdm(total=len(futures), desc="Processing images")

        try:
            for future in concurrent.futures.as_completed(futures):
                filename = futures[future]  # 获取正在处理的文件名

                if should_stop.is_set():
                    for f in futures:
                        f.cancel()
                    print("Batch processing was stopped by the user.")
                    break

                try:
                    result = future.result()
                except Exception as e:
                    result = (filename, f"An exception occurred: {e}")
                    print(f"An exception occurred while processing {filename}: {e}")
                results.append(result)
                progress.update(1)


        finally:
            progress.close()
            executor.shutdown(wait=False)

    results = f"Total checked images: {len(results)}"
    return results

# api
def switch_API(api, state):
    def is_connection():
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1)
            print("API has started.")
            return True
        except (socket.timeout, ConnectionRefusedError):
            return False
    if api[:3] == 'GPT' or api[:4] == "qwen":
        if is_connection():
            requests.post(f"http://127.0.0.1:8000/v1/close")
        key = saved_api_key
        url = saved_api_url
        time_out = 10
        if api[:4] == "qwen" and url.endswith("/v1/services/aigc/multimodal-generation/generation"):
            mod = qwen_api_switch(api)
        else:
            mod = 'GPT4V'
        s_state = mod

    elif api[:3] == 'Cog' or api[:4] == "moon":
        if is_connection():
            if state != api:
                requests.post(f"http://127.0.0.1:8000/v1/{api}")
        else:
            API_command = f'python openai_api.py --mod {api}'
            subprocess.Popen(API_command,shell=True)
            while True:
                if is_connection():
                    break
                else:
                    print("Retrying...")
                    time.sleep(2)

        key = ""
        url = "http://127.0.0.1:8000/v1/chat/completions"
        time_out = 300
        s_state = api

    return key, url, time_out, s_state

# UI界面
with gr.Blocks(title="GPT4V captioner") as demo:
    gr.Markdown("### Image Captioning with GPT-4-Vision API / 使用 GPT-4-Vision API 进行图像打标")

    with gr.Row():
        api_key_input = gr.Textbox(label="API Key", placeholder="Enter your GPT-4-Vision API Key here", type="password",
                                   value=saved_api_key)
        api_url_input = gr.Textbox(label="API URL", value=saved_api_url or "https://api.openai.com/v1/chat/completions",
                                   placeholder="Enter the GPT-4-Vision API URL here")
        quality_choices = [
            ("Auto / 自动", "auto"),
            ("High Detail - More Expensive / 高细节-更贵", "high"),
            ("Low Detail - Cheaper / 低细节-更便宜", "low")
        ]
        quality = gr.Dropdown(choices=quality_choices, label="Image Quality / 图片质量", value="auto")
        timeout_input = gr.Number(label="Timeout (seconds) / 超时时间（秒）", value=10, step=1)

    prompt_input = gr.Textbox(label="Prompt / 打标需求",
                              value="As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. Prioritize the tags by relevance. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and camera specifications, and any other tags crucial for the image. When tagging photos of people, include specific details like gender, nationality, attire, actions, pose, expressions, accessories, makeup, composition type, age, etc. For other image categories, apply appropriate and common descriptive tags as well. Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. Your tags should be accurate, non-duplicative, and within a 20-75 word count range. These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. Tags should be comma-separated. Exceptional tagging will be rewarded with $10 per image.",
                              placeholder="Enter a descriptive prompt",
                              lines=5)

    with gr.Accordion("Prompt Saving / 提示词存档", open=False):
        def update_textbox(prompt):
            return prompt
        saved_pro = get_prompts_from_csv()
        saved_prompts_dropdown = gr.Dropdown(label="Saved Prompts / 提示词存档", choices=saved_pro, type="value",interactive=True)
        with gr.Row():
            save_prompt_button = gr.Button("Save Prompt / 保存提示词")
            delete_prompt_button = gr.Button("Delete Prompt / 删除提示词")
            load_prompt_button = gr.Button("Load Prompt / 读取到输入框")

        save_prompt_button.click(save_prompt, inputs=prompt_input,outputs=[saved_prompts_dropdown])
        delete_prompt_button.click(delete_prompt, inputs=saved_prompts_dropdown, outputs=[saved_prompts_dropdown])
        load_prompt_button.click(update_textbox, inputs=saved_prompts_dropdown, outputs=prompt_input)

    with gr.Tab("Image Process / 图片处理"):

        with gr.Tab("Image Zip / 图像预压缩"):
            with gr.Row():
                folder_path_input = gr.Textbox(
                    label="Image Folder Path / 图像文件夹路径",
                    placeholder="Enter the folder path containing images / 输入包含图像的文件夹路径"
                )
                process_images_button = gr.Button("Process Images / 压缩图像")

            with gr.Row():
                # Add a Markdown component to display the warning message
                gr.Markdown("""
            ⚠ **Warning / 警告**: This preprocessing process will resize and compress all image files into jpg format with a total pixel count ≤ 1024×1024 while maintaining the original aspect ratio, ensuring that both dimensions are multiples of 32. **Please make sure to backup your original files before processing!** This procedure can reduce the size of the training set, help to speed up the labeling process, and decrease the time taken to cache latents to disk during training.

            本预处理过程将会在保持原图长宽比情况下，把所有图像文件裁剪压缩为总像素≤1024×1024的jpg文件，并且长宽像素均为32的倍数。**请务必在处理前备份源文件！**该过程可以缩小训练集体积，有助于加快打标速度，并缩短训练过程中的Cache latents to disk时间。
                """)

            with gr.Row():
                image_processing_output = gr.Textbox(
                    label="Image Processing Output / 图像处理输出",
                    lines=3
                )

            process_images_button.click(process_images_in_folder,
                inputs=[folder_path_input],
                outputs=[image_processing_output])

        with gr.Tab("Single Image / 单图处理"):
            with gr.Row():
                image_input = gr.Image(type='filepath', label="Upload Image / 上传图片")
                single_image_output = gr.Textbox(label="Caption Output / 标签输出")
            with gr.Row():
                single_image_submit = gr.Button("Caption Single Image / 图片打标", variant='primary')

        with gr.Tab("Batch Image / 多图批处理"):
            with gr.Row():
                batch_dir_input = gr.Textbox(label="Batch Directory / 批量目录",
                                             placeholder="Enter the directory path containing images for batch processing")
            with gr.Row():
                batch_process_submit = gr.Button("Batch Process Images / 批量处理图像", variant='primary')
            with gr.Row():
                batch_output = gr.Textbox(label="Batch Processing Output / 批量输出")
                file_handling_mode = gr.Radio(
                    choices=["overwrite/覆盖", "prepend/前置插入", "append/末尾追加", "skip/跳过"],
                    value="overwrite/覆盖",
                    label="If a caption file exists: / 如果已经存在打标文件: "
                )
            with gr.Row():
                stop_button = gr.Button("Stop Batch Processing / 停止批量处理")
                stop_button.click(stop_batch_processing, inputs=[], outputs=batch_output)

        with gr.Tab("Failed File Screening / 打标失败文件筛查"):
            folder_input = gr.Textbox(label="Folder Input / 文件夹输入", placeholder="Enter the directory path")
            keywords_input = gr.Textbox(placeholder="Enter keywords, e.g., sorry,error / 请输入检索关键词，例如：sorry,error",
                                        label="Keywords (optional) / 检索关键词（可选）")
            run_button = gr.Button("Run Script / 运行脚本", variant='primary')
            output_area = gr.Textbox(label="Script Output / 脚本输出")

            run_button.click(fn=run_script, inputs=[folder_input, keywords_input], outputs=output_area)

    with gr.Tab("Extra Function / 额外功能"):

        gr.Markdown("""
                    以下功能基于CogVLM开发（GPT4未经测试），极力推荐使用CogVLM-vqa以达到最佳效果。\n
                    This function is developed based on CogVLM (GPT4 not tested), and it is strongly recommended to use CogVLM-vqa for optimal results.
                    """)

        with gr.Tab("Watermark Detection / 批量水印检测"):
            with gr.Row():
                detect_batch_dir_input = gr.Textbox(label="Image Directory / 图片目录",
                                                    placeholder="Enter the directory path containing images for batch processing")
            with gr.Row():
                watermark_dir = gr.Textbox(label="Watermark Detected Image Directory / 检测到水印的图片目录",
                                           placeholder="Enter the directory path to move/copy detected images")
                detect_file_handling_mode = gr.Radio(choices=["move/移动", "copy/复制"], value="move/移动",
                                                     label="If watermark is detected / 如果图片检测到水印 ")
            with gr.Row():
                batch_detect_submit = gr.Button("Batch Detect Images / 批量检测图像", variant='primary')
            with gr.Row():
                detect_batch_output = gr.Textbox(label="Output / 结果")
            with gr.Row():
                detect_stop_button = gr.Button("Stop Batch Processing / 停止批量处理")
                detect_stop_button.click(stop_batch_processing, inputs=[], outputs=detect_batch_output)
        with gr.Tab("WD1.4 Tag Polishing / WD1.4 标签润色"):
            gr.Markdown("""
                    使用WD1.4对图片进行打标后，在上方prompt中使用“Describe this image in a very detailed manner and refer these prompt tags:{大括号里替换为放置额外tags文件的目录，会自动读取和图片同名txt。比如 D:\ abc\}”\n
                    After marking the image using WD1.4, enter the prompt in the “” marks in the prompt box above.
                        “Replace this with the directory where additional tags files are placed, which will automatically read the txt file with the same name as the image. For example, D: \ abc\}”
                    """)
        with gr.Tab("Image filtering / 图片筛选"):
            gr.Markdown("""
                        使用自定义规则筛选图片，将回答中包含或不包含对应词的图片放入对应规则的文件夹中。输出目录默认在源目录下的classify_output文件夹下。\n
                        Use custom rules to filter images. Place images containing or not containing corresponding words in the corresponding rule folder in the answer. Output Directory default in source directory \classify_output.
                        """)
            with gr.Row():
                classify_output = gr.Textbox(label="Output / 结果")
                classify_button = gr.Button("Run / 开始", variant='primary')
                classify_stop_button = gr.Button("Stop Batch Processing / 停止批量处理")
            with gr.Row():
                classify_dir = gr.Textbox(label="Input Image Directory / 输入图片目录",placeholder="Enter the directory path")
                classify_output_dir = gr.Textbox(label="Output Directory / 输出目录", placeholder="Default source directory / 默认源目录")
                classify_handling_mode = gr.Radio(label="If meets / 如果符合",choices=["move/移动", "copy/复制"], value="move/移动")

            rule_inputs = []
            for i in range(1,11):
                with gr.Row():
                    rule_type = gr.Dropdown(label="Rule / 规则类型", choices=["","Involve / 包含", "Exclude / 不包含"], value="")
                    rule_input = gr.Textbox(label="Custom / 自定义", placeholder="Enter the words you need to filter / 输入你需要筛选的词")
                    rule_inputs.extend([rule_type, rule_input])

    def caption_image(api_key, api_url, prompt, image, quality, timeout):
        if image:
            return process_single_image(api_key, prompt, api_url, image, quality, timeout)

    def batch_process(api_key, api_url, prompt, batch_dir, file_handling_mode, quality, timeout):
        process_batch_images(api_key, prompt, api_url, batch_dir, file_handling_mode, quality, timeout)
        return "Batch processing complete. Captions saved or updated as '.txt' files next to images."

    def batch_detect(api_key, api_url, prompt, batch_dir, detect_file_handling_mode, quality, timeout, watermark_dir):
        results = process_batch_watermark_detection(api_key, prompt, api_url, batch_dir, detect_file_handling_mode,
                                                    quality, timeout,watermark_dir)
        return results

    single_image_submit.click(caption_image,
                              inputs=[api_key_input, api_url_input, prompt_input, image_input, quality, timeout_input],
                              outputs=single_image_output)
    batch_process_submit.click(batch_process,
                               inputs=[api_key_input, api_url_input, prompt_input, batch_dir_input,
                                       file_handling_mode, quality, timeout_input],
                               outputs=batch_output)
    batch_detect_submit.click(batch_detect,
                              inputs=[api_key_input, api_url_input, prompt_input, detect_batch_dir_input,
                                      detect_file_handling_mode, quality, timeout_input, watermark_dir],
                              outputs=detect_batch_output)

    classify_button.click(classify_images,
                          inputs=[api_key_input, api_url_input, quality, prompt_input, timeout_input,
                                  classify_handling_mode, classify_dir, classify_output_dir] + rule_inputs,
                          outputs=classify_output)
    classify_stop_button.click(stop_batch_processing,inputs=[],outputs=classify_output)

    with gr.Tab("Tag Manage / 标签处理"):

        with gr.Row():
            folder_path_input = gr.Textbox(label="Folder Path / 文件夹路径",
                                           placeholder="Enter folder path / 在此输入文件夹路径")
            top_n_input = gr.Number(label="Top N Tags / Top N 标签", value=100)
            translate_tags_input = gr.Radio(label="Translate Tags to Chinese / 翻译标签",
                                            choices=["GPT-3.5 translation / GPT3.5翻译",
                                                     "Free translation / 免费翻译",
                                                     "No translation / 不翻译"],
                                            value="No translation / 不翻译")
            process_tags_button = gr.Button("Process Tags / 处理标签", variant='primary')
            output_message = gr.Textbox(label="Output Message / 输出信息", interactive=False)

        with gr.Row():
            tags_to_remove_input = gr.Textbox(label="Tags to Remove / 删除标签",
                                              placeholder="Enter tags to remove, separated by commas / 输入要删除的标签，用逗号分隔",
                                              lines=3)
            tags_to_replace_input = gr.Textbox(label="Tags to Replace / 替换标签",
                                               placeholder="Enter tags to replace in 'old_tag:new_tag' format, separated by commas / 输入要替换的标签，格式为 '旧标签:新标签'，用逗号分隔",
                                               lines=3)
            new_tag_input = gr.Textbox(label="Add New Tag / 添加新标签",
                                       placeholder="Enter a new tag to add / 输入一个新标签以添加", lines=3)
            insert_position_input = gr.Radio(label="New Tag Insert Position / 新标签插入位置",
                                             choices=["Start / 开始", "End / 结束", "Random / 随机"],
                                             value="Start / 开始")

        with gr.Row():
            wordcloud_output = gr.Image(label="Word Cloud / 词云")
            tag_counts_output = gr.Dataframe(label="Top Tags / 高频标签",
                                             headers=["Tag Name", "Frequency", "Chinese Translation"],
                                             interactive=True)  # 修改 Dataframe 组件以显示三列

        with gr.Row():
            network_graph_output = gr.Image(label="Network Graph / 网络图")

        process_tags_button.click(process_tags,
                                  inputs=[folder_path_input, top_n_input, tags_to_remove_input,
                                        tags_to_replace_input, new_tag_input, insert_position_input,
                                        translate_tags_input, api_key_input, api_url_input], # 新增翻译复选框
                                  outputs=[tag_counts_output, wordcloud_output, network_graph_output, output_message])


    # API Config
    with gr.Tab("API Config / API配置"):
        # CogVLM一键
        with gr.Accordion("Local Model / 使用本地模型", open=True):
            with gr.Row():
                gr.Markdown("""
            ⚠ **Warning / 警告**: 
            This is the API configuration page. To use local model, you need to configure environment and download it.
                            **Moondream** model **size is about 22g+**, and it takes a long time, Please confirm that the disk space is sufficient.Please confirm that your GPU has sufficient graphics memory ***(approximately 6g)*** 
                            **CogVLM**, you need to configure environment and download it, which is **approximately 35g+** in size and takes a long time ***(really, really long)***. 
                            After installation and download, the total space occupied is about ***40g+***. Please confirm that the disk space is sufficient.
                            In addition, in terms of model selection, the vqa model performs better but slower, while the chat model is faster but slightly weaker.
                            Please confirm that your GPU has sufficient graphics memory ***(approximately 14g ±)*** when using CogVLM
                        
            此为API配置页面，使用本地模型需要配置相关环境并下载模型，
                            ***moondream***模型大小约为**22g+**需要较长时间，请确认磁盘空间充足。显存需求约为6g，请确认自己的显卡有足够的显存。
                            ***CogVLM***大小约为**35g+**，需要较长时间 **(真的很长)**。安装以及下载完成后，总占用空间约为40g+，请确认磁盘空间充足。
                            模型选择上，vqa模型效果更好但是更慢，chat模型更快但是效果略弱。使用CogVLM请确认自己的显卡有足够的显存 ***(约14g±)***
            """)
            with gr.Row():
                detecter_output = gr.Textbox(label="Check Env / 环境检测", interactive=False)
                detect_button = gr.Button("Check / 检查", variant='primary')
            with gr.Row():
                models_select = gr.Radio(label="Choose Models / 选择模型", choices=["moondream","vqa", "chat"], value="moondream")
                acceleration_select = gr.Radio(label="Choose Default Plz / 选择是否国内加速(如果使用国内加速,请关闭魔法上网)", choices=["CN", "default"],
                                               value="CN")
                download_button = gr.Button("Download Models / 下载模型", variant='primary')
                install_button = gr.Button("Install / 安装", variant='primary')

        # API配置
        mod_list = [
            "GPT4V",
            "qwen-vl-plus",
            "qwen-vl-max",
            "moondream",
            "Cog-vqa",
            "Cog-chat"
            ]
        with gr.Row():
            switch_select = gr.Dropdown(label="Choose API / 选择API", choices=mod_list, value="GPT4V")
            A_state = gr.Textbox(label="API State / API状态", interactive=False, value=mod_default)
            switch_button = gr.Button("Switch / 切换", variant='primary')
            set_default = gr.Button("Set as default / 设为默认", variant='primary')


        detect_button.click(detecter, outputs=detecter_output)
        download_button.click(downloader, inputs=[models_select, acceleration_select],
                              outputs=detecter_output)
        install_button.click(installer, outputs=detecter_output)
        switch_button.click(switch_API, inputs=[switch_select, A_state],
                            outputs=[api_key_input, api_url_input, timeout_input, A_state])
        set_default.click(save_state, inputs=[switch_select, api_key_input, api_url_input], outputs=A_state)

    gr.Markdown(
        "### Developers: [Jiaye](https://civitai.com/user/jiayev1),&nbsp;&nbsp;[LEOSAM 是只兔狲](https://civitai.com/user/LEOSAM),&nbsp;&nbsp;[SleeeepyZhou](https://civitai.com/user/SleeeepyZhou),&nbsp;&nbsp;[Fok](https://civitai.com/user/fok3827)&nbsp;&nbsp;|&nbsp;&nbsp;Welcome everyone to add more new features to this project.")

# 启动参数
def get_args():
    parser = argparse.ArgumentParser(description='GPT4V-Image-Captioner启动参数')
    parser.add_argument("--port", type=int, default="8848", help="占用端口，默认8848")
    parser.add_argument("--listen", action='store_true', help="打开远程连接，默认关闭")
    parser.add_argument("--share", action='store_true', help="打开gradio共享，默认关闭")
    return parser.parse_args()

args = get_args()

if __name__ == "__main__":
    threading.Thread(target=lambda: switch_API(mod_default, 'GPT')).start()
    demo.launch(
        server_name="0.0.0.0" if args.listen else None,
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )
