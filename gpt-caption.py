# Import the necessary libraries
import base64
import requests
import json
import os
import gradio as gr
from tqdm import tqdm

# Function to get the saved API key and URL
def get_saved_api_details():
    settings_file = 'api_settings.json'
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        return settings.get('api_key', ''), settings.get('api_url', '')
    return '', ''

# Function to save the API key and URL
def save_api_details(api_key, api_url):
    settings = {
        'api_key': api_key,
        'api_url': api_url
    }
    with open('api_settings.json', 'w') as f:
        json.dump(settings, f)

# Function to run the OpenAI API query using cURL
def run_openai_api(image_path, prompt, api_key, api_url):
    # Convert image to base64
    image_file = open(image_path, "rb").read()
    image_base64 = base64.b64encode(image_file).decode('utf-8')
    
    # Construct the 'data' payload for the request
    data = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 300
    }
    
    # Set up the headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Make the request
    response = requests.post(api_url, headers=headers, json=data)
    
    if response.status_code != 200:
        # If the request failed, return the error
        return f"Error: API request failed with status {response.status_code}\n{response.text}"

    # Parse the JSON response
    try:
        response_data = response.json()
        
        # Check if 'error' in response_data for API specific errors
        if 'error' in response_data:
            return f"API error: {response_data['error']['message']}"
        
        # Extracting the caption from the response_data
        caption = response_data["choices"][0]["message"]["content"]
        return caption
    except Exception as e:
        # Log the full output for debugging and return a friendly message
        return f"Failed to parse the API response: {e}\n{response.text}"

# Function to process a single image
def process_single_image(api_key, prompt, api_url, image_path):
    save_api_details(api_key, api_url)
    #image.save("/tmp/image_to_process.jpg")
    caption = run_openai_api(image_path, prompt, api_key, api_url)
    print(caption)
    return caption

# Updated batch processing function
def process_batch_images(api_key, prompt, api_url, image_dir):
    save_api_details(api_key, api_url)
    results = []

    # Prepare the list of image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Use tqdm to create the progress bar
    for filename in tqdm(image_files, desc='Processing images'):
        image_path = os.path.join(image_dir, filename)
        caption = run_openai_api(image_path, prompt, api_key, api_url)
        print(caption)
        # Remove original file extension and replace with '.txt'
        base_filename = os.path.splitext(image_path)[0]
        caption_filename = f"{base_filename}.txt"
        # Save the caption
        with open(caption_filename, 'w') as file:
            file.write(caption)
        results.append((filename, caption_filename))
    return results

saved_api_key, saved_api_url = get_saved_api_details()

# Gradio interface setup
with gr.Blocks(title="GPT4V captioner") as demo:
    gr.Markdown("Image Captioning with OpenAI's GPT-4-Vision API")
    with gr.Row():
        api_key_input = gr.Textbox(label="API Key", placeholder="Enter your OpenAI API Key here", type="password", value=saved_api_key)
        api_url_input = gr.Textbox(label="API URL", value=saved_api_url or "https://api.openai.com/v1/chat/completions", placeholder="Enter the OpenAI API URL here")
        
    with gr.Tab("Single Image Processing"):
        with gr.Row():
            image_input = gr.Image(type='filepath', label="Upload Image")
        with gr.Row():
            single_image_submit = gr.Button("Caption Single Image", variant='primary')
        single_image_output = gr.Textbox(label="Single Image Caption Output")
    
    with gr.Tab("Batch Image Processing"):
        with gr.Row():
            # batch_api_key_input = gr.Textbox(label="API Key", placeholder="Enter your OpenAI API Key here", type="password", value=saved_api_key, visible=False)
            # batch_api_url_input = gr.Textbox(label="API URL", value=saved_api_url or "https://api.openai.com/v1/chat/completions", placeholder="Enter the OpenAI API URL here", visible=False)
            # batch_prompt_input = gr.Textbox(label="Prompt", value="What’s in this image?", placeholder="Enter the same prompt used for single image", visible=False)
            batch_dir_input = gr.Textbox(label="Batch Directory", placeholder="Enter the directory path containing images for batch processing")
        with gr.Row():
            batch_process_submit = gr.Button("Batch Process Images", variant='secondary')
        batch_output = gr.Textbox(label="Batch Processing Output")
    prompt_input = gr.Textbox(label="Prompt", value="What’s in this image?", placeholder="Enter a descriptive prompt")
    def batch_process(api_key, api_url, prompt, batch_dir):
        process_batch_images(api_key, prompt, api_url, batch_dir) 
        return "Batch processing complete. Captions saved as '.txt' files next to images."
    
    def caption_image(api_key, api_url, prompt, image):
        if image:  # Processing a single image
            return process_single_image(api_key, prompt, api_url, image)

    single_image_submit.click(caption_image, inputs=[api_key_input, api_url_input, prompt_input, image_input], outputs=single_image_output)
    batch_process_submit.click(batch_process, inputs=[api_key_input, api_url_input, prompt_input, batch_dir_input], outputs=batch_output)

demo.launch()