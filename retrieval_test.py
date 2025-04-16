#！-*- coding: utf-8 -*-
# SimBERT 相似度任务测试
# 基于LCQMC语料

import numpy as np
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding

import pandas as pd

maxlen = 32

# bert配置
config_path = 'chinese_simbert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_simbert_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])


def load_data(filename):
    data = pd.read_parquet(filename)
    data = data.values
    
    D = []
    for line in data:
        text1, text2, label = line
        D.append((text1, text2))
    return D


# 测试相似度效果
data = load_data('datasets/lcqmc/validation-00000-of-00001-ae04bea7d65ea894.parquet')
a_token_ids, b_token_ids = [], []
texts = []

for d in data:
    token_ids = tokenizer.encode(d[0], max_length=maxlen)[0]
    a_token_ids.append(token_ids)
    token_ids = tokenizer.encode(d[1], max_length=maxlen)[0]
    b_token_ids.append(token_ids)
    texts.extend(d)

a_token_ids = sequence_padding(a_token_ids)
b_token_ids = sequence_padding(b_token_ids)
a_vecs = encoder.predict([a_token_ids, np.zeros_like(a_token_ids)],
                         verbose=True)
b_vecs = encoder.predict([b_token_ids, np.zeros_like(b_token_ids)],
                         verbose=True)

a_vecs = a_vecs / (a_vecs**2).sum(axis=1, keepdims=True)**0.5
b_vecs = b_vecs / (b_vecs**2).sum(axis=1, keepdims=True)**0.5


# 测试全量检索能力
vecs = np.concatenate([a_vecs, b_vecs], axis=1).reshape(-1, 768)


def most_similar(text, topn=10):
    """检索最相近的topn个句子
    """
    token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
    vec = encoder.predict([[token_ids], [segment_ids]])[0]
    vec /= (vec**2).sum()**0.5
    sims = np.dot(vecs, vec)
    return [(texts[i], sims[i]) for i in sims.argsort()[::-1][:topn]]

if __name__=="__main__":
    result = most_similar(u'怎么开初婚未育证明', 20)
    print(result)
    