#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : 2020-04-13 21:46
# @Author  : zhangjingbo
# @mail    : jingbo.zhang@yooli.com
# @File    : run_iter.py
# @Description:  循环训练
# @Software: PyCharm
import os
import random
import torch
import warnings

import torch.optim as optim

from pytorch_recepit.机器人翻译.common_params import hidden_size, batch_size, output_n_layers, input_n_layers, dropout
from pytorch_recepit.机器人翻译.data_prepare import batch2TrainData
from pytorch_recepit.机器人翻译.data_process import prepare_Data
from pytorch_recepit.机器人翻译.rnn_model import RNNEncoder, LuongAttnDecoderRNN
from pytorch_recepit.机器人翻译.translateRun import train

warnings.filterwarnings("ignore")


def trainIters(model_name, pairs, encoder, encoder_hidden, decoder, encoder_optimizer, decoder_optimizer,
               encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip,
               input_lang, output_lang):
    # 为每次迭代加载batches
    training_batches = [batch2TrainData(input_lang, output_lang, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # 初始化
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    losses_list = []
    # 训练循环
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # 从batch中提取字段
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # 使用batch运行训练迭代
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder, encoder_hidden,
                     decoder, encoder_optimizer, decoder_optimizer, batch_size, clip, input_lang, output_lang)
        print_loss += loss

        # 打印进度
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration,
                                                                                          iteration / n_iteration * 100,
                                                                                          print_loss_avg))
            print_loss = 0
            losses_list.append(print_loss)
        # 保存checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name,
                                     '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'fra': encoder.state_dict(),
                'eng': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'plot_losses': losses_list
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


if __name__ == '__main__':
    model_name = 'eng2fra_model'
    input_lang, output_lang, pairs = prepare_Data('eng', 'fra', 'char', True)
    encoder = RNNEncoder(input_lang.n_words, hidden_size, input_n_layers, dropout)
    decoder = LuongAttnDecoderRNN('general', hidden_size, output_lang.n_words, output_n_layers, dropout)
    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())
    save_dir = '/Users/zhangjingbo/Workspaces/PycharmProjects/data-analysis/pytorch_recepit/机器人翻译/'
    clip = 50.0
    n_iteration = 4000
    print_every = 10
    save_every = 500
    encoder_hidden = torch.zeros(2 * input_n_layers, batch_size, hidden_size, device='cpu')
    trainIters(model_name, pairs, encoder, encoder_hidden, decoder, encoder_optimizer, decoder_optimizer,
               input_n_layers,
               output_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, input_lang,
               output_lang)
