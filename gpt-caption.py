import gradio as gr
import os
import shutil
import threading

import concurrent.futures
from tqdm import tqdm

import subprocess
import time
import requests
import socket
import platform
from huggingface_hub import snapshot_download

from lib.Img_Processing import process_images_in_folder, run_script
from lib.Tag_Processor import modify_file_content, process_tags
from lib.GPT_Prompt import get_prompts_from_csv, save_prompt, delete_prompt
from lib.Api_Utils import run_openai_api, save_api_details, get_api_details


os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
saved_api_key, saved_api_url = get_api_details()

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

def process_batch_watermark_detection(api_key, prompt, api_url, image_dir, detect_file_handling_mode, quality, timeout,
                                      watermark_dir):
    should_stop.clear()
    save_api_details(api_key, api_url)
    results = []
    prompt = 'Is image have watermark'

    supported_image_formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif')
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(supported_image_formats):
                image_files.append(os.path.join(root, file))

    def process_image(filename, detect_file_handling_mode, watermark_dir):
        image_path = os.path.join(image_dir, filename)
        caption = run_openai_api(image_path, prompt, api_key, api_url, quality, timeout)

        if caption.startswith("Error:") or caption.startswith("API error:"):
            return "error"

        # EOI是cog迷之误判？
        if 'Yes,' in caption and '\'EOI\'' not in caption:
            if detect_file_handling_mode == "copy/复制":
                shutil.copy(filename, watermark_dir)
            if detect_file_handling_mode == "move/移动":
                shutil.move(filename, watermark_dir)

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

        process_tags_button.click(
            process_tags,
            inputs=[
                folder_path_input, top_n_input, tags_to_remove_input,
                tags_to_replace_input, new_tag_input, insert_position_input,
                translate_tags_input,  # 新增翻译复选框
                api_key_input, api_url_input
            ],
            outputs=[tag_counts_output, wordcloud_output, network_graph_output, output_message]
        )

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

        process_images_button.click(
            fn=process_images_in_folder,
            inputs=[folder_path_input],
            outputs=[image_processing_output]
        )

    with gr.Tab("Watermark Detection / 批量水印检测"):
        gr.Markdown("""
                本功能完全是基于CogVLM开发（GPT4未经测试），极力推荐使用CogVLM-vqa以达到最佳效果。\n
                This function is fully developed based on CogVLM (GPT4 not tested), and it is strongly recommended to use CogVLM-vqa for optimal results.
                """)
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



    # CogVLM一键
    with gr.Tab("CogVLM Config / CogVLM配置"):
        with gr.Row():
            gr.Markdown("""
        ⚠ **Warning / 警告**: This is the CogVLM configuration page. To use CogVLM, you need to download it, which is approximately 35g+ in size and takes a long time (really, really long).
                        In addition, in terms of model selection, the vqa model performs better but slower, while the chat model is faster but slightly weaker.
                        Please confirm that your GPU has sufficient graphics memory (approximately 12g ±) when using CogVLM

        此为CogVLM配置页面，使用CogVLM需要配置相关环境并下载模型，大小约为35g+，需要较长时间（真的很长）。模型选择上，vqa模型效果更好但是更慢，chat模型更快但是效果略弱。
        使用CogVLM请确认自己的显卡有足够的显存（约12g±）
            """)

        with gr.Row():
            models_select = gr.Radio(label="Choose Models / 选择模型", choices=["vqa", "chat"], value="vqa")
            acceleration_select = gr.Radio(label="Choose Default Plz / 选择是否国内加速", choices=["CN", "default"],
                                           value="CN")
            download_button = gr.Button("Download Models / 下载模型", variant='primary')
            install_button = gr.Button("Install / 安装", variant='primary')

        with gr.Row():
            switch_select = gr.Radio(label="Choose API / 选择API", choices=["GPT", "Cog"], value="GPT")
            models_switch = gr.Radio(label="Choose Use Cog Models / 选择使用的Cog模型", choices=["vqa", "chat"],
                                     value="vqa")
            A_state = gr.Textbox(label="API State / API状态", interactive=False, value="GPT")
            switch_button = gr.Button("Switch / 切换", variant='primary')


        def download_snapshot(model_type, acceleration):
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


        def install_cog(acceleration):
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
            except subprocess.CalledProcessError as e:
                print(f'Error: {e}')


        def switch_API(api, cogmod, state):
            if api == 'GPT':
                key = saved_api_key
                url = saved_api_url
                time_out = 10
                s_state = "GPT"

            elif api == 'Cog':

                def is_connection():
                    try:
                        socket.create_connection(("127.0.0.1", 8000), timeout=1)
                        print("API has started.")
                        return True
                    except (socket.timeout, ConnectionRefusedError):
                        return False

                if is_connection():
                    if state[-3:] != cogmod:
                        requests.post(f"http://127.0.0.1:8000/v1/{cogmod}")
                else:
                    API_command = f'python cog_openai_api.py --model {cogmod}'
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
                s_state = f"Cog-{cogmod}"

            return key, url, time_out, s_state


        download_button.click(download_snapshot, inputs=[models_select, acceleration_select], outputs=download_button)
        install_button.click(install_cog, inputs=[acceleration_select], outputs=install_button)
        switch_button.click(switch_API, inputs=[switch_select, models_switch, A_state],
                            outputs=[api_key_input, api_url_input, timeout_input, A_state])


    def batch_process(api_key, api_url, prompt, batch_dir, file_handling_mode, quality, timeout):
        process_batch_images(api_key, prompt, api_url, batch_dir, file_handling_mode, quality, timeout)
        return "Batch processing complete. Captions saved or updated as '.txt' files next to images."


    def batch_detect(api_key, api_url, prompt, batch_dir, detect_file_handling_mode, quality, timeout, watermark_dir):
        results = process_batch_watermark_detection(api_key, prompt, api_url, batch_dir, detect_file_handling_mode,
                                                    quality, timeout,
                                                    watermark_dir)
        return results


    def caption_image(api_key, api_url, prompt, image, quality, timeout):
        if image:
            return process_single_image(api_key, prompt, api_url, image, quality, timeout)


    single_image_submit.click(caption_image,
                              inputs=[api_key_input, api_url_input, prompt_input, image_input, quality, timeout_input],
                              outputs=single_image_output)
    batch_process_submit.click(
        batch_process,
        inputs=[api_key_input, api_url_input, prompt_input, batch_dir_input, file_handling_mode, quality,
                timeout_input],
        outputs=batch_output
    )
    batch_detect_submit.click(
        batch_detect,
        inputs=[api_key_input, api_url_input, prompt_input, detect_batch_dir_input, detect_file_handling_mode, quality,
                timeout_input, watermark_dir],
        outputs=detect_batch_output
    )

    gr.Markdown(
        "### Developers: [Jiaye](https://civitai.com/user/jiayev1),&nbsp;&nbsp;[LEOSAM 是只兔狲](https://civitai.com/user/LEOSAM),&nbsp;&nbsp;[SleeeepyZhou](https://civitai.com/user/SleeeepyZhou),&nbsp;&nbsp;[Fok](https://civitai.com/user/fok3827)&nbsp;&nbsp;|&nbsp;&nbsp;Welcome everyone to add more new features to this project.")

if __name__ == "__main__":
    demo.launch(server_port=8848)
