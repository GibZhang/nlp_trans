#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : 2020-04-12 11:08
# @Author  : zhangjingbo
# @mail    : jingbo.zhang@yooli.com
# @File    : data_prepare.py
# @Description: 将处理后的文本资料，转换为模型的输入输出张量
# @Software: PyCharm
import torch
from pytorch_recepit.机器人翻译.data_process import prepare_Data
import itertools
import random
import numpy as np
from pytorch_recepit.机器人翻译.common_params import EOS_token, device, PAD_token



def indexesFromSentence(lang, sentence):
    sentence = sentence.lower()
    return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def sentenceFromIndexes(lang, indexTensor):
    seq_length, batch_size = indexTensor.shape
    for i in range(batch_size):
        sentence_array = indexTensor[:, i].numpy()
        sentence = [lang.index2word[idx] for idx in sentence_array]
        print(sentence)


# zip 对数据进行合并了，相当于行列转置了
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


# 记录 PAD_token的位置为0， 其他的为1
def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# 返回填充前（加入结束index EOS_token做标记）的长度 和 填充后的输入序列张量
def inputVar(l, lang):
    indexes_batch = [indexesFromSentence(lang, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# 返回填充前（加入结束index EOS_token做标记）最长的一个长度 和 填充后的输入序列张量, 和 填充后的标记 mask
def outputVar(l, lang):
    indexes_batch = [indexesFromSentence(lang, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# 返回给定batch对的所有项目
def batch2TrainData(input_lang, output_lang, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, input_lang)
    output, mask, max_target_len = outputVar(output_batch, output_lang)
    return inp, lengths, output, mask, max_target_len


