#！-*- coding: utf-8 -*-
# SimBERT 相似度任务测试
# 基于LCQMC语料

import numpy as np
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding

import json
import os
import subprocess


def most_similar(text, database, topn=10):
    """检索最相近的topn个句子
    """
    
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
    with open(database, "r") as f:
        data = json.load(f)

    for wav_id, info in data.items():
        desc = info["description"]
        token_id = tokenizer.encode(desc, max_length=maxlen)[0]
        token_ids.append(token_id)
        items.append((wav_id, desc))

    token_ids = sequence_padding(token_ids)
    vecs = encoder.predict([token_ids, np.zeros_like(token_ids)], verbose=True)

    vecs = vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
    vecs = vecs.reshape(-1, 768)
    
    # 当前 prompt
    token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
    vec = encoder.predict([[token_ids], [segment_ids]])[0]
    vec /= (vec**2).sum()**0.5
    sims = np.dot(vecs, vec)
    topn_result = [(items[i], sims[i]) for i in sims.argsort()[::-1][:topn]]
    
    gender = None
    if "女" in text:
        gender = "女"
    elif "男" in text:
        gender = "男"
    
    if gender is not None:
        ret = []
        for _, (item, sim) in enumerate(topn_result):
            if gender in item[1]:
                ret.append(topn_result[_])
        return ret
    else:
        return topn_result

def main():
    # 参数设置
    database = "test.json"
    query_text = "中年女子的音调低沉，音量适中，语速慢慢地。"
    top_n = 20
    wav_base_path = "/home/group3/wsl/datasets/cut_wav"
    f5_path = "/home/group3/wsl/F5-TTS/src/f5_tts/infer"
    
    # 查询最相似的结果
    result = most_similar(query_text, database, top_n)
    print(f"retrieval result:\n{result}\n")
    
    if len(result) > 0:
        # 获取最佳匹配的音频ID和路径
        audio_id = result[0][0][0]
        audio_path = os.path.join(wav_base_path, f"{audio_id}.wav")
        
        # 获取转录文本
        with open(database, "r") as f:
            data = json.load(f)
        transcription = data[audio_id]["transcription"]

        gen_text = "Some text you want TTS model generate for you."
        
        cmd = f"cd {f5_path}\npython infer_cli.py --model F5TTS_v1_Base --ref_audio {audio_path} --ref_text '{transcription}' --gen_text '{gen_text}'"
        print(f"cmd:\n{cmd}")


if __name__ == "__main__":
    main()