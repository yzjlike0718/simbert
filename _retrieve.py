#！-*- coding: utf-8 -*-
# SimBERT 相似度任务测试
# 基于LCQMC语料

import numpy as np
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding

import json

def most_similar(text, topn=10):
    """检索最相近的topn个句子
    """
    gender = None
    if "女" in text:
        gender = "女"
    elif "男" in text:
        gender = "男"
    print(f"gender: {gender}")
    
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

    # 设定最大长度
    maxlen = 128

    token_ids = []
    items = []
    # 读取数据
    # with open("/home/group3/workspace/wsl/datasets/dbpara_test_4k_simplified.json", "r") as f:
    with open("test.json", "r") as f:
        data = json.load(f)

    for wav_id, desc in data.items():
        token_id = tokenizer.encode(desc, max_length=maxlen)[0]
        token_ids.append(token_id)
        items.append((wav_id, desc))

    token_ids = sequence_padding(token_ids)
    vecs = encoder.predict([token_ids, np.zeros_like(token_ids)], verbose=True)
    vecs = vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
    vecs = vecs.reshape(-1, 768)
    
    token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
    vec = encoder.predict([[token_ids], [segment_ids]])[0]
    vec /= (vec**2).sum()**0.5
    sims = np.dot(vecs, vec).argsort()[::-1]
    print(sims[:5])
    if gender is not None:
        ret = []
        for _, sim in enumerate(sims):
            if gender in items[_][1]:
                ret.append((items[_], sim))
                if len(ret) == topn:
                    break
        return ret
    else:
        return [(items[i], sims[i]) for i in sims[:topn]]

if __name__=="__main__":
    result = most_similar(u'中年男子的音调高亢，音量适中，语速慢慢地。', 20)
    print(result)
    