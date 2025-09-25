# """
#     Adapter: Liu Martin
# """
import os
from tools.errors import FINISH_STOP, FINISH_LENGTH
from openai import OpenAI, BadRequestError, APITimeoutError

# Instructions for starting the localhost vllm server to interface with openai sdk
# uv venv --python 3.12 --seed
# source .venv/bin/activate
# uv pip install --pre vllm==0.10.1+gptoss \
#     --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
#     --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
#     --index-strategy unsafe-best-match
#
# vllm serve openai/gpt-oss-20b

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="Empty"  # vLLM accepts any non-empty string
)


def process_with_openai_gpt_oss_20B(request, max_tokens=None, temperature=0.0):  
    if max_tokens is None:
        max_tokens = 8192
    return openai_call(request, "openai/gpt-oss-20b", temperature=temperature, max_tokens=max_tokens)


def openai_call(request, model, temperature=0.0, max_tokens=None):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": request['prompt']}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )    
    except (BadRequestError, APITimeoutError) as e:
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        raise e

    if response.choices[0].finish_reason == "length":
        finish_reason = FINISH_LENGTH
    elif response.choices[0].finish_reason == "stop":
        finish_reason = FINISH_STOP
    else:
        return None

    return response.choices[0].message.content, {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,  
        "thinking_tokens": 0,
        "finish_reason": finish_reason
    }


# import os
# from tools.errors import FINISH_STOP, FINISH_LENGTH

# from openai_harmony import (
#     HarmonyEncodingName,
#     load_harmony_encoding,
#     Conversation,
#     Message,
#     Role,
#     SystemContent,
#     DeveloperContent,
# )

# from vllm import LLM, SamplingParams


# def process_with_openai_gpt_oss_20B(request, max_tokens=None, temperature=0.0):
#     if max_tokens is None:
#         max_tokens = 32768 # should be plenty
#     return vllm_call(request, "gpt-oss-20b", temperature=temperature, max_tokens=max_tokens)


# # Could move this to a new file when we generalize.
# def vllm_call(request, model, temperature=0.0, max_tokens=None):
#     return None