from transformers import MarianMTModel, MarianTokenizer
from tools.errors import FINISH_STOP, FINISH_LENGTH


def process_with_opus_en_ja(request, max_tokens=None, temperature=0.0):
    if max_tokens is None:
        max_tokens = 8192
    return opus_call(request, "Helsinki-NLP/opus-mt-en-jap", temperature=temperature, max_tokens=max_tokens)


def process_with_opus_en_de(request, max_tokens=None, temperature=0.0):
    if max_tokens is None:
        max_tokens = 8192
    return opus_call(request, "Helsinki-NLP/opus-mt-en-de", temperature=temperature, max_tokens=max_tokens)


# Egyptian Arabic processed with the ar model, and language label >>arz<<
def process_with_opus_en_ar(request, max_tokens=None, temperature=0.0):
    language_label = ">>arz<<"
    request["prompt"] = f"{language_label} {request["prompt"]}"

    if max_tokens is None:
        max_tokens = 8192
    return opus_call(request, "Helsinki-NLP/opus-mt-en-ar", temperature=temperature, max_tokens=max_tokens)


# Bengali processed with the inc model, and language label >>ben<<
def process_with_opus_en_bn(request, max_tokens=None, temperature=0.0):
    language_label = ">>ben<<"
    request["prompt"] = f"{language_label} {request["prompt"]}"

    if max_tokens is None:
        max_tokens = 8192
    return opus_call(request, "Helsinki-NLP/opus-mt-en-inc", temperature=temperature, max_tokens=max_tokens)


def opus_call(request, model, temperature=0.0, max_tokens=None):
    # Load the pre-trained model and tokenizer
    model_name = model
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    # Source text
    source_texts = [request["prompt"]]

    #  Tokenize the input texts
    inputs = tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True)

    # Generate translations
    translated_ids = model.generate(inputs["input_ids"])

    # Decode the generated tokens to get the translated text
    translated_texts = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)

    return translated_texts[0], {
        "input_tokens": 0,
        "output_tokens": 0,  
        "thinking_tokens": 0,
        "finish_reason": FINISH_STOP
    }

