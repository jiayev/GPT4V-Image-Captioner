import gc
import time
import requests
import base64
import uvicorn
import argparse

import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer, PreTrainedModel, PreTrainedTokenizer, \
TextIteratorStreamer, CodeGenTokenizerFast as Tokenizer

from contextlib import asynccontextmanager
from loguru import logger
from typing import List, Literal, Union, Tuple, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from PIL import Image
from io import BytesIO

import os
import re
from threading import Thread
from moondream import Moondream, detect_device

# 请求
class TextContent(BaseModel):
    type: Literal["text"]
    text: str
class ImageUrl(BaseModel):
    url: str
class ImageUrlContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl
ContentItem = Union[TextContent, ImageUrlContent]
class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessageInput]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # Additional parameters
    repetition_penalty: Optional[float] = 1.0

# 响应
class ChatMessageResponse(BaseModel):
    role: Literal["assistant"]
    content: str = None
    name: Optional[str] = None
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessageResponse
class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None

# 图片输入处理
def process_img(input_data):
    if isinstance(input_data, str):
        # URL
        if input_data.startswith("http://") or input_data.startswith("https://"):
            response = requests.get(input_data)
            image_data = response.content
            pil_image = Image.open(BytesIO(image_data)).convert('RGB')
        # base64
        elif input_data.startswith("data:image/"):
            base64_data = input_data.split(",")[1]
            image_data = base64.b64decode(base64_data)
            pil_image = Image.open(BytesIO(image_data)).convert('RGB')
        # img_path
        else:
            pil_image = Image.open(input_data)
    # PIL
    elif isinstance(input_data, Image.Image):
        pil_image = input_data
    else:
        raise ValueError("data type error")

    return pil_image

# 历史消息处理
def process_history_and_images(messages: List[ChatMessageInput]) -> Tuple[
    Optional[str], Optional[List[Tuple[str, str]]], Optional[List[Image.Image]]]:
    formatted_history = []
    image_list = []
    last_user_query = ''

    for i, message in enumerate(messages):
        role = message.role
        content = message.content

        if isinstance(content, list):  # text
            text_content = ' '.join(item.text for item in content if isinstance(item, TextContent))
        else:
            text_content = content

        if isinstance(content, list):  # image
            for item in content:
                if isinstance(item, ImageUrlContent):
                    image_url = item.image_url.url
                    image = process_img(image_url)
                    image_list.append(image)

        if role == 'user':
            if i == len(messages) - 1:  # last message
                last_user_query = text_content
            else:
                formatted_history.append((text_content, ''))
        elif role == 'assistant':
            if formatted_history:
                if formatted_history[-1][1] != '':
                    assert False, f"the last query is answered. answer again. {formatted_history[-1][0]}, {formatted_history[-1][1]}, {text_content}"
                formatted_history[-1] = (formatted_history[-1][0], text_content)
            else:
                assert False, f"assistant reply before user"
        else:
            assert False, f"unrecognized role: {role}"

    return last_user_query, formatted_history, image_list

@torch.inference_mode()
# Moondrean推理
def generate_stream_moondream(params: dict):
    global model, tokenizer

    # 输入处理
    def chat_history_to_prompt(history):
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += f"Question: {old_query}\n\nAnswer: {response}\n\n"
        return prompt

    messages = params["messages"]
    prompt, formatted_history, image_list = process_history_and_images(messages)
    history = chat_history_to_prompt(formatted_history)
    # 只处理最后一张图
    img = image_list[-1]

    # 构建输入
    '''
    answer_question(
            self,
            image_embeds,
            question,
            tokenizer,
            chat_history="",
            result_queue=None,
            **kwargs,
        )
    '''
    image_embeds = model.encode_image(img)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    gen_kwargs = {
            "image_embeds": image_embeds,
            "question": prompt,
            "tokenizer": tokenizer,
            "chat_history": history,
            "result_queue": None,
            "streamer": streamer,
        }

    thread = Thread(
        target=model.answer_question,
        kwargs=gen_kwargs,
    )

    input_echo_len = 0
    total_len = 0
    # 启动推理
    thread.start()
    buffer = ""
    for new_text in streamer:
        clean_text = re.sub("<$|END$", "", new_text)
        buffer += clean_text
        yield {
            "text": buffer.strip("<END"),
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
                },
            }
    generated_ret ={
        "text": buffer.strip("<END"),
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
            },
        }
    yield generated_ret

# Moondrean单次响应
def generate_moondream(params: dict):
    for response in generate_stream_moondream(params):
        pass
    return response


@torch.inference_mode()
# CogVLM推理
def generate_stream_cogvlm(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    """
    Generates a stream of responses using the CogVLM model in inference mode.
    It's optimized to handle continuous input-output interactions with the model in a streaming manner.
    """
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    query, history, image_list = process_history_and_images(messages)

    logger.debug(f"==== request ====\n{query}")

    #  only can slove the latest picture
    input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history,
                                                        images=[image_list[-1]])
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]],
    }
    if 'cross_images' in input_by_model and input_by_model['cross_images']:
        inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

    input_echo_len = len(inputs["input_ids"][0])
    streamer = TextIteratorStreamer(tokenizer=tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "top_p": top_p,
        'streamer': streamer,
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    total_len = 0
    generated_text = ""
    with torch.no_grad():
        model.generate(**inputs, **gen_kwargs)
        for next_text in streamer:
            generated_text += next_text
            yield {
                "text": generated_text,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": total_len - input_echo_len,
                    "total_tokens": total_len,
                },
            }
    ret = {
        "text": generated_text,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
    }
    yield ret

# CogVLM单次响应
def generate_cogvlm(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):

    for response in generate_stream_cogvlm(model, tokenizer, params):
        pass
    return response


# 流式响应
async def predict(model_id: str, params: dict):
    return "no stream"

torch.set_grad_enabled(False)
# 生命周期管理器，结束清显存
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
app = FastAPI(lifespan=lifespan)
# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 对话路由
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    # 检查请求
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
    )

    # 流式响应
    if request.stream:
        generate = predict(request.model, gen_params)
        return

    # 单次响应
    if STATE_MOD == "cog":
        response = generate_cogvlm(model, tokenizer, gen_params)
    else:
        response = generate_moondream(gen_params)
    usage = UsageInfo()
    message = ChatMessageResponse(
        role="assistant",
        content=response["text"],
    )
    logger.debug(f"==== message ====\n{message}")
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
    )
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion", usage=usage)

# 模型切换路由配置
STATE_MOD = "moon"
MODEL_PATH = ""

# 模型加载
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_mod(model_input, mod_type):
    global model, tokenizer, language_processor_version
    if mod_type == "cog":
        tokenizer_path = os.environ.get("TOKENIZER_PATH", 'lmsys/vicuna-7b-v1.5')
        tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            signal_type=language_processor_version
        )
        if 'cuda' in DEVICE:
            model = AutoModelForCausalLM.from_pretrained(
                model_input,
                trust_remote_code=True,
                load_in_4bit=True,
                torch_dtype=torch_type,
                low_cpu_mem_usage=True
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_input,
                trust_remote_code=True
            ).float().to(DEVICE).eval()
    else:
        device, dtype = detect_device()
        model = Moondream.from_pretrained(model_input).to(device=device, dtype=dtype).eval()
        tokenizer = Tokenizer.from_pretrained(model_input)

@app.post("/v1/Cog-vqa")
async def switch_vqa():
    global model, STATE_MOD, mod_vqa, language_processor_version
    STATE_MOD = "cog"
    del model
    model = None
    language_processor_version = "chat_old"
    load_mod(mod_vqa, STATE_MOD)

@app.post("/v1/Cog-chat")
async def switch_chat():
    global model, STATE_MOD, mod_chat, language_processor_version
    STATE_MOD = "cog"
    del model
    model = None
    language_processor_version = "chat"
    load_mod(mod_chat, STATE_MOD)

@app.post("/v1/moondream")
async def switch_moon():
    global model, STATE_MOD, mod_moon
    STATE_MOD = "moon"
    del model
    model = None
    load_mod(mod_moon, STATE_MOD)

# 关闭
@app.post("/v1/close")
async def close():
    global model
    del model
    model = None

gc.collect()

parser = argparse.ArgumentParser()
parser.add_argument("--mod", type=str, default="moondrean")
args = parser.parse_args()
mod = args.mod

mod_vqa = './models/cogagent-vqa-hf'
mod_chat = './models/cogagent-chat-hf'
mod_moon = './models/moondream'

'''
mod_list = [
    "moondrean",
    "Cog-vqa",
    "Cog-chat"
    ]
'''

if mod == "Cog-vqa":
    STATE_MOD = "cog"
    MODEL_PATH = mod_vqa
    language_processor_version = "chat_old"
elif mod == "Cog-chat":
    STATE_MOD = "cog"
    MODEL_PATH = mod_chat
    language_processor_version = "chat"
elif mod == "moondream":
    STATE_MOD = "moon"
    MODEL_PATH = mod_moon

if __name__ == "__main__":
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch_type = torch.bfloat16
    else:
        torch_type = torch.float16

    print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

    load_mod(MODEL_PATH, STATE_MOD)

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
