#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : 2020-04-13 21:54
# @Author  : zhangjingbo
# @mail    : jingbo.zhang@yooli.com
# @File    : evaluate.py
# @Description: 
# @Software: PyCharm
import os
import warnings

import torch.nn as nn

from pytorch_recepit.机器人翻译.data_prepare import sentenceFromIndexes
from pytorch_recepit.机器人翻译.data_process import prepare_Data, normalizeString
from pytorch_recepit.机器人翻译.rnn_model import RNNEncoder, LuongAttnDecoderRNN

warnings.filterwarnings("ignore")
from pytorch_recepit.机器人翻译.common_params import *
from pytorch_recepit.机器人翻译.data_prepare import indexesFromSentence


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length, hidden):
        # 通过编码器模型转发输入
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length, hidden)
        # 准备编码器的最终隐藏层作为解码器的第一个隐藏输入
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # 使用SOS_token初始化解码器输入
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # 初始化张量以将解码后的单词附加到
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # 一次迭代地解码一个词tokens
        for _ in range(max_length):
            # 正向通过解码器
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 获得最可能的单词标记及其softmax分数
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # 记录token和分数
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # 准备当前令牌作为下一个解码器输入（添加维度）
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # 返回收集到的词tokens和分数
        return all_tokens, all_scores


def evaluate(searcher, input_lang, output_lang, sentence, hidden, max_length=MAX_LENGTH):
    import re
    sentence = normalizeString(sentence)
    ### 格式化输入句子作为batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(input_lang, sentence)]
    # 创建lengths张量
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 转置batch的维度以匹配模型的期望
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # 使用合适的设备
    input_batch = input_batch.to(device)
    sentenceFromIndexes(input_lang, input_batch)
    lengths = lengths.to(device)
    # 用searcher解码句子
    tokens, scores = searcher(input_batch, lengths, max_length, hidden)
    # indexes -> words
    decoded_words = [output_lang.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(input_lang, output_lang, searcher, hidden, input_sentence):
    # 评估句子
    output_words = evaluate(searcher, input_lang, output_lang, input_sentence, hidden, 20)
    # 格式化和打印回复句
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    print('Trans:', ' '.join(output_words))


if __name__ == '__main__':
    model_name = 'eng2fra_model'
    input_lang, output_lang, pairs = prepare_Data('eng', 'fra', 'char', True)
    encoder = RNNEncoder(input_lang.n_words, hidden_size, input_n_layers, dropout)
    decoder = LuongAttnDecoderRNN('general', hidden_size, output_lang.n_words, output_n_layers, dropout)
    save_dir = '/Users/zhangjingbo/Workspaces/PycharmProjects/data-analysis/pytorch_recepit/机器人翻译/'
    seacher = GreedySearchDecoder(encoder, decoder)
    hidden = torch.zeros(2 * input_n_layers, 1, hidden_size, device='cpu')
    model_save = torch.load(os.path.join(save_dir, model_name,
                                         '{}-{}_{}'.format(input_n_layers, output_n_layers,
                                                           hidden_size)) + '/4000_checkpoint.tar')
    encoder.load_state_dict(model_save['fra'])
    encoder.eval()
    decoder.load_state_dict(model_save['eng'])
    decoder.eval()
    sentenses_eval = [['I am a shy boy.', 'Je suis un garçon timide.'],
                      ['I am a student.', 'Je suis étudiant.'],
                      ['I am a teacher.', 'Je suis professeur.'],
                      ['I am a tourist.', 'Je suis touriste.'],
                      ["They're coming.", "Elles arrivent."],
                      ["They're coming.", "Ils sont en train d'arriver."],
                      ["They're coming.", "Elles sont en train d'arriver."],
                      ["They're idiots.", "Ils sont idiots."],
                      ["They're inside.", "Ils sont à l'intérieur."], ]
    for input_sentence in sentenses_eval:
        evaluateInput(input_lang, output_lang, seacher, hidden, input_sentence[1])
