#authors : Urvish and Mobassir

import logging
import requests
import os
import io
import glob
import time
import json


import tensorflow as tf
import transformers
from transformers import AutoTokenizer
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# redefining transformers custom model

MODEL = 'jplu/tf-xlm-roberta-large'
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Class map for the sentiment
CLASS_MAP = {0: 'Negative', 1:'Positive'}

# Encoding function
def regular_encode(texts, tokenizer, maxlen=192):
    """Generates tokens based on the tokenizer used.

    Args:
        texts (array): array of text strings
        tokenizer (tokenizer): Transformer tokenizer to encode the text
        maxlen (int, optional): Max length of the sequence. Defaults to 512.

    Returns:
        array: Tokenized text using the tokenizer for desired model.
    """
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        #truncation=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )

    return np.array(enc_di['input_ids'])

# Deserialize the Invoke request body into an object we can perform prediction on
def input_handler(data, context):
    logger.info('Deserializing the input data.')
    if context.request_content_type == 'application/json':
        # pass through json (assumes it's correctly formed)
        d = data.read().decode('utf-8')
        d = pd.Series(d)
        print(d, type(d))
        enc_ids = regular_encode(d, tokenizer).tolist()[0]
        print(len(enc_ids))
        print(np.array(enc_ids).shape)
        json_obj = json.dumps({'instances': [enc_ids]})
        print(json_obj)
        return json_obj
    else:
        raise ValueError("Wrong input format given")

#mapping predicted Encoded values to labels
def fun(x, dix):
    return dix[x]

MapEncodedLabel = np.vectorize(fun)

# Serialize the prediction result into the desired response content type
def output_handler(data, context):
   
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))
    
    response_content_type = context.accept_header
    prediction = json.loads(data.content)
    print(prediction)
    #preds = np.argmax(np.array(prediction["predictions"]), axis=1)

    preds = prediction["predictions"][0][0]
    if (preds >= 0.5):
        preds = 1
    else:
        preds = 0
    print(preds)
    idx = MapEncodedLabel(preds, CLASS_MAP).tolist()
    print(idx)
    json_obj = json.dumps({"predictions" : idx})
    return json_obj, response_content_type
