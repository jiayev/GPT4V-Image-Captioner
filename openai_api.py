import gc
import io
import json
import time
import requests
import base64
import uvicorn
import argparse

import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer, PreTrainedModel, PreTrainedTokenizer, \
TextIteratorStreamer, CodeGenTokenizerFast as Tokenizer
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from omnilmm.utils import disable_torch_init
from omnilmm.model.omnilmm import OmniLMMForCausalLM
from omnilmm.model.utils import build_transform
from omnilmm.train.train_utils import omni_preprocess

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

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, process_images, get_model_name_from_path,
                            tokenizer_image_token)
from llava.model import *


from llava.model.builder import load_pretrained_model

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=['lm_head'])

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

def eval_vila_model(args, model, tokenizer, image_processor):
    # read json file
    with open(args.test_json_path) as f:
        all_test_cases = json.load(f)

    result_list = []
    print(len(all_test_cases["test_cases"]))

    for test_case in all_test_cases["test_cases"]:
        # read images first
        image_file_list = test_case["image_paths"]
        image_list = [
            Image.open(os.path.join(args.test_image_path, image_file)).convert("RGB") for image_file in image_file_list
        ]
        image_tensor = process_images(image_list, image_processor, model.config)

        # image_tokens = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

        for i in range(len(test_case["QAs"])):
            query = test_case["QAs"][i]["question"]
            query_text = query

            if 1:
                # query = query.replace("<image>", image_tokens)
                if len(image_list) < 3:
                    conv = conv_templates["vicuna_v1"].copy()
                else:
                    conv = conv_templates["vicuna_v1_nosys"].copy()
                conv.append_message(conv.roles[0], query)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
            else:
                conv = conv_templates[args.conv_mode].copy()
                if not "<image>" in query:
                    assert "###" not in query  # single query
                    query = image_tokens + "\n" + query  # add <image>
                    query_list = [query]
                else:
                    query_list = query.split("###")
                    assert len(query_list) % 2 == 1  # the last one is from human

                    new_query_list = []
                    for idx, query in enumerate(query_list):
                        if "<image>" in query:
                            assert idx % 2 == 0  # only from human
                            # assert query.startswith("<image>")
                        # query = query.replace("<image>", image_tokens)
                        new_query_list.append(query)
                    query_list = new_query_list

                for idx, query in enumerate(query_list):
                    conv.append_message(conv.roles[idx % 2], query)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

            print("%" * 10 + " " * 5 + "VILA Response" + " " * 5 + "%" * 10)

            # inputs = tokenizer([prompt])
            inputs = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX)
            input_ids = torch.as_tensor(inputs).cuda().unsqueeze(0)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # outputs = run_llava.process_outputs(args, model, tokenizer, input_ids, image_tensor, stopping_criteria, stop_str)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=0.7,
                    # top_p=args.top_p,
                    # num_beams=args.num_beams,
                    max_new_tokens=512,
                    # use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()

            print(f"Question: {query_text}")
            print(f"VILA output: {outputs}")
            print(f'Expected output: {test_case["QAs"][i]["expected_answer"]}')

            result_list.append(
                dict(question=query_text, output=outputs, expected_output=test_case["QAs"][i]["expected_answer"])
            )
    return result_list

def init_omni_lmm(model_path):
    torch.backends.cuda.matmul.allow_tf32 = True
    disable_torch_init()
    model_name = os.path.expanduser(model_path)
    print(f'Load omni_lmm model and tokenizer from {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=2048)

    if False:
        # model on multiple devices for small size gpu memory (Nvidia 3090 24G x2) 
        with init_empty_weights():
            model = OmniLMMForCausalLM.from_pretrained(model_name, tune_clip=True, torch_dtype=torch.bfloat16)
        model = load_checkpoint_and_dispatch(model, model_name, dtype=torch.bfloat16, 
                    device_map="auto",  no_split_module_classes=['Eva','MistralDecoderLayer', 'ModuleList', 'Resampler']
        )
    else:
        model = OmniLMMForCausalLM.from_pretrained(
            model_name, tune_clip=True, torch_dtype=torch.bfloat16
        ).to(device='cuda', dtype=torch.bfloat16)

    image_processor = build_transform(
        is_train=False, input_size=model.model.config.image_size, std_mode='OPENAI_CLIP')

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    assert mm_use_im_start_end

    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IM_END_TOKEN], special_tokens=True)


    vision_config = model.model.vision_config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = model.model.config.num_query

    return model, image_processor, image_token_len, tokenizer

def expand_question_into_multimodal(question_text, image_token_len, im_st_token, im_ed_token, im_patch_token):
    if '<image>' in question_text[0]['content']:
        question_text[0]['content'] = question_text[0]['content'].replace(
            '<image>', im_st_token + im_patch_token * image_token_len + im_ed_token)
    else:
        question_text[0]['content'] = im_st_token + im_patch_token * \
            image_token_len + im_ed_token + '\n' + question_text[0]['content']
    return question_text

def wrap_question_for_omni_lmm(question, image_token_len, tokenizer):
    question = expand_question_into_multimodal(
        question, image_token_len, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN)

    conversation = question
    data_dict = omni_preprocess(sources=[conversation],
                                  tokenizer=tokenizer,
                                  generation=True)

    data_dict = dict(input_ids=data_dict["input_ids"][0],
                     labels=data_dict["labels"][0])
    return data_dict



class OmniLMM12B:
    def __init__(self, model_path) -> None:
        model, img_processor, image_token_len, tokenizer = init_omni_lmm(model_path)
        self.model = model
        self.image_token_len = image_token_len
        self.image_transform = img_processor
        self.tokenizer = tokenizer
        self.model.eval()

    def decode(self, image, input_ids):
        with torch.inference_mode():
            output = self.model.generate_vllm(
                input_ids=input_ids.unsqueeze(0).cuda(),
                images=image.unsqueeze(0).half().cuda(),
                temperature=0.6,
                max_new_tokens=1024,
                # num_beams=num_beams,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                repetition_penalty=1.1,
                top_k=30,
                top_p=0.9,
            )

            response = self.tokenizer.decode(
                output.sequences[0], skip_special_tokens=True)
            response = response.strip()
            return response

    def chat(self, input):
        try:
            image = Image.open(io.BytesIO(base64.b64decode(input['image']))).convert('RGB')
        except Exception as e:
            return f"Image decode error, {e}"

        msgs = json.loads(input['question'])
        input_ids = wrap_question_for_omni_lmm(
            msgs, self.image_token_len, self.tokenizer)['input_ids']
        input_ids = torch.as_tensor(input_ids)
        #print('input_ids', input_ids)
        image = self.image_transform(image)

        out = self.decode(image, input_ids)

        return out
        

def img2base64(file_name):
    with open(file_name, 'rb') as f:
        encoded_string = base64.b64encode(f.read())
        return encoded_string

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

# Vila推理
def generate_vila(model, tokenizer, image_processor, context_len, params: dict):
    messages = params["messages"]
    query, history, image_list = process_history_and_images(messages)
    msgs = history
    msgs.append({'role': 'user', 'content': query})
    image = image_list[-1]
    image_tensor = process_images([image], image_processor, model.config)
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX)
    input_ids = torch.as_tensor(inputs).cuda().unsqueeze(0)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
            do_sample=True,
            temperature=0.6,
            top_p=0.7,
            max_new_tokens=512,
            stopping_criteria=[stopping_criteria],
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    return outputs

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

# OmniLMM单次响应
def generate_omni(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    messages = params["messages"]
    query, history, image_list = process_history_and_images(messages)
    msgs = history
    msgs.append({'role': 'user', 'content': query})
    image = image_list[-1]
    # image is a PIL image
    buffer = BytesIO()
    image.save(buffer, format="JPEG")  # You can adjust the format as needed
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read())
    image_base64_str = image_base64.decode("utf-8")
    input = {'image': image_base64_str, 'question': json.dumps(msgs)}
    response = model.chat(input)
    print(response)
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
    global model, tokenizer, image_processor, context_len

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
    elif STATE_MOD == "moon":
        response = generate_moondream(gen_params)
    elif STATE_MOD == "omni":
        response = generate_omni(model, tokenizer, gen_params)
    elif STATE_MOD == "vila":
        response = generate_vila(model, tokenizer, image_processor, context_len, gen_params)
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
    elif mod_type == "moon":
        device, dtype = detect_device()
        model = Moondream.from_pretrained(model_input).to(device=device, dtype=dtype).eval()
        tokenizer = Tokenizer.from_pretrained(model_input)
    elif mod_type == "omni":
        device, dtype = detect_device()
        model, tokenizer = OmniLMM12B(model_input), None
    elif mod_type == "vila":
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_input, get_model_name_from_path(model_input), None, load_4bit=True)

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

@app.post("/v1/Omni")
async def switch_omni():
    global model, STATE_MOD, mod_omni
    STATE_MOD = "omni"
    del model
    model = None
    load_mod(mod_omni, STATE_MOD)

@app.post("/v1/VILA")
async def switch_vila():
    global model, STATE_MOD, mod_vila
    STATE_MOD = "vila"
    del model
    model = None
    load_mod(mod_vila, STATE_MOD)

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
mod_omni = './models/OmniLMM-12B'
mod_vila = './models/VILA1.5-13b'

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
elif mod == "OmniLMM":
    STATE_MOD = "omni"
    MODEL_PATH = mod_omni
elif mod == "VILA":
    STATE_MOD = "vila"
    MODEL_PATH = mod_vila

if __name__ == "__main__":
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch_type = torch.bfloat16
    else:
        torch_type = torch.float16

    print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

    load_mod(MODEL_PATH, STATE_MOD)

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
