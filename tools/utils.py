"""
    Adaptation of translation collection repository.
    Adapter: Liu Martin

    Adapted from https://github.com/wmt-conference/wmt-collect-translations
"""

import re
import os
import time
import glob
import logging
import traceback
import hashlib
import pandas as pd
import diskcache as dc
from tqdm import tqdm
from collections import defaultdict

from tools.providers.openai import process_with_openai_gpt_oss_20B
from tools.errors import FINISH_LENGTH, FINISH_STOP, ERROR_UNSUPPORTED_LANGUAGE


SYSTEMS = {
    'GPT-OSS-20B': process_with_openai_gpt_oss_20B
}

def check_paragraph_alignment(source_text, translated_text):
    """
    Check if the number of paragraphs in the source and translated text are the same. This is requirement of GenMT 2025
    """
    source_paragraphs = source_text.split('\n\n')
    translated_paragraphs = translated_text.split('\n\n')

    if len(source_paragraphs) != len(translated_paragraphs):
        return False
    return True


def remove_tripple_quotes(text):
    # check if there are exactly two occurences of ``` in the text
    if text.count("```") == 2:
        # get only the text inbetween the tripple quotes
        text = text.split("```")[1]

    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    return text


def _process_document_level(system_name, request, translation_granularity):
    if translation_granularity == 'document-level':
        request['prompt'] = f"{request['prompt_instruction']}\n\n{request['segment']}"
    elif translation_granularity == 'document-level-wrapped':
        request['prompt'] = f"{request['prompt_instruction']}\n\n```{request['segment']}```"
    elif translation_granularity == 'document-level-html':
        segment = request['segment'].replace('\n\n', '\n<br>\n\n')
        if "Please translate the following" in request['prompt_instruction']:
            instruction = request['prompt_instruction'].replace('Please translate the following', 'Keep HTML tags in the answer. Please translate the following')
        else:
            raise ValueError("Prompt instruction should contain 'Please translate the following'")
        request['prompt'] = f"{instruction}\n\n{segment}"

    answer = SYSTEMS[system_name](request)

    if answer is None or answer == ERROR_UNSUPPORTED_LANGUAGE:
        return answer

    answer, tokens = answer

    if translation_granularity == 'document-level-wrapped':
        answer = remove_tripple_quotes(answer)
    elif translation_granularity == 'document-level-html':
        answer = re.sub(r'\n*\s*<br>\s*\n*', '<br>', answer)
        answer = re.sub(r'\n{2,}', '\n', answer)
        answer = answer.replace('<br>', '\n\n')
    
    return answer, tokens


# TODO: in WMT26, remove line-level granularity
def _process_line_level(system_name, request):
    answers = []
    tokens = {}
    seg_request = request.copy()
    highest_temperature = 0.0
    for sentence in request['segment'].split('\n'):
        seg_request['prompt'] = f"{request['prompt_instruction']}\n\n{sentence}"
        seg_request['segment'] = sentence

        for temperature in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            answer = SYSTEMS[system_name](seg_request, temperature=temperature)
            if temperature > highest_temperature:
                highest_temperature = temperature
            
        if answer is None:
            return answer, 0.0

        translated_sentence, sentence_tokens = answer
        translated_sentence = re.sub(r'\n{1,}', ' ', translated_sentence)
        answers.append(translated_sentence.strip('\n'))
        # add tokens to the dictionary
        if sentence_tokens is not None:
            for key, value in sentence_tokens.items():
                if key == "finish_reason":
                    continue
                if key not in tokens:
                    tokens[key] = 0
                tokens[key] += value
    
    tokens['finish_reason'] = FINISH_STOP
    return ('\n'.join(answers), tokens), highest_temperature


def _process_paragraph_level(system_name, request, translation_granularity='paragraph-level'):
    answers = []
    tokens = {}
    seg_request = request.copy()
    highest_temperature = 0.0
    for paragraph in request['segment'].split('\n\n'):
        seg_request['prompt'] = f"{request['prompt_instruction']}\n\n{paragraph}"
        seg_request['segment'] = paragraph

        answer = SYSTEMS[system_name](seg_request)
        if answer[1]['finish_reason'] == FINISH_LENGTH:
            # there are few long paragraphs in testsuites, use sentence-level only for those
            answer, temperature = _process_line_level(system_name, seg_request)
            if translation_granularity == 'paragraph-level':
                translation_granularity = f'line-level'

            if temperature > highest_temperature:
                highest_temperature = temperature
                translation_granularity = f'line-level-{highest_temperature}'

        if answer is None or answer[1]['finish_reason'] == FINISH_LENGTH:
            return answer, translation_granularity
        
        translated_paragraph, paragraph_tokens = answer
        translated_paragraph = re.sub(r'\n{2,}', '\n', translated_paragraph)
        answers.append(translated_paragraph.strip())
        # add tokens to the dictionary
        if paragraph_tokens is not None:
            for key, value in paragraph_tokens.items():
                if key == "finish_reason":
                    continue
                if key not in tokens:
                    tokens[key] = 0
                tokens[key] += value
    tokens['finish_reason'] = FINISH_STOP
    return ('\n\n'.join(answers), tokens), translation_granularity


def _request_system(system_name, request):
    attempt_document_level = True
    # it was already failing on max_tokens before the last try
    for translation_granularity in ['document-level', 'document-level-wrapped', 'document-level-html', 'paragraph-level']:
        if not attempt_document_level and translation_granularity != 'paragraph-level':
            continue

        if "document-level" in translation_granularity:
            answer = _process_document_level(system_name, request, translation_granularity)
        elif translation_granularity == 'paragraph-level':
            answer, translation_granularity = _process_paragraph_level(system_name, request, translation_granularity)

        if answer is None:
            print(f"System {system_name} returned None for doc_id {request['doc_id']} with translation granularity {translation_granularity}")
            continue
        if answer[1]['finish_reason'] == FINISH_LENGTH:
            attempt_document_level = False
            continue
        if answer == ERROR_UNSUPPORTED_LANGUAGE:
            raise ValueError(ERROR_UNSUPPORTED_LANGUAGE)

        answer, tokens = answer

        answer = answer.strip()
        if check_paragraph_alignment(request['segment'], answer):
            return {
                'doc_id': request['doc_id'],
                'translation': answer,
                'translation_granularity': translation_granularity,
                "tokens": tokens
            }
        else:
            print(f"Paragraph alignment failed for {system_name} on doc_id {request['doc_id']} with {translation_granularity}.")

    return None


# TODO: WMT25 - track temperature and reasoning traces in metadata

def collect_answers(blindset, system_name):
    cache = dc.Cache(f'cache/{system_name}', expire=None, size_limit=int(10e10), cull_limit=0, eviction_policy='none')

    answers = []
    unsupported_languages = []
    for _, row in tqdm(blindset.iterrows(), total=len(blindset), desc=system_name):
        # completely skip unsupported languages
        if (row['src_lang'], row['tgt_lang']) in unsupported_languages:
            continue

        request = {
            'doc_id': row['doc_id'],
            'source_language': row['src_lang'],
            'target_language': row['tgt_lang'],
            'segment': row['src_text'],
            'prompt_instruction': row['prompt_instruction']
        }
        # create hash merging all the information in the request
        hashid = f"{request['doc_id']}_{request['source_language']}_{request['target_language']}_{request['segment']}_{request['prompt_instruction']}"
        hashid = hashlib.md5(hashid.encode('utf-8')).hexdigest()

        # None represent problem in the translation that was originally skipped
        if hashid not in cache or cache[hashid] is None:
            try:
                cache[hashid] = _request_system(system_name, request)
            except Exception as e:
                if str(e) == ERROR_UNSUPPORTED_LANGUAGE:
                    unsupported_languages.append((row['src_lang'], row['tgt_lang']))
                    continue
                logging.error(f"Error processing {request['doc_id']} with {system_name}: {e}")
                logging.error(traceback.format_exc())
                cache[hashid] = None

        if cache[hashid] is not None:
            answers.append(cache[hashid])
        else:
            paragraphs = len(request['segment'].split('\n\n'))
            answers.append({
                'doc_id': request['doc_id'],
                'translation': '\n\n'.join(["FAILED"] * paragraphs), # if everything fails, there is nothing we can do
                'translation_granularity': None,
            })
        # mandatory fields for submission to OCELoT
        answers[-1]['dataset_id'] = "wmttest2025"
        answers[-1]['tgt_lang'] = row['tgt_lang']
        answers[-1]['hypothesis'] = answers[-1].pop('translation')

    return answers