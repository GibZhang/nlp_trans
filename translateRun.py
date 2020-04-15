#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : 2020-04-12 16:05
# @Author  : zhangjingbo
# @mail    : jingbo.zhang@yooli.com
# @File    : translateRun.py
# @Description: 翻译模型训练
# @Software: PyCharm
import random

import torch

from pytorch_recepit.机器人翻译.data_prepare import sentenceFromIndexes

device = 'gpu' if torch.cuda.is_available() else 'cpu'
PAD_token = 0
SOS_token = 1
EOS_token = 2
teacher_forcing_ratio = 0.5
MAX_LENGTH = 20


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, encoder_hidden, decoder,
          encoder_optimizer, decoder_optimizer, batch_size, clip, input_lang, output_lang):
    # 用于检查转移后的张量是否有问题
    # sentenceFromIndexes(input_lang, input_variable)
    # sentenceFromIndexes(output_lang, target_variable)
    # 零化梯度
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 设置设备选项
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # 初始化变量
    loss = 0
    print_losses = []
    n_totals = 0

    # 正向传递编码器
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths, encoder_hidden)

    # 创建初始解码器输入（从每个句子的SOS令牌开始）
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # 将初始解码器隐藏状态设置为编码器的最终隐藏状态
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # 确定我们是否此次迭代使用`teacher forcing`
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # 通过解码器一次一步地转发一批序列
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: 下一个输入是当前的目标
            decoder_input = target_variable[t].view(1, -1)
            # 计算并累计损失
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: 下一个输入是解码器自己的当前输出
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # 计算并累计损失
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # 执行反向传播
    loss.backward()

    # 剪辑梯度：梯度被修改到位
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # 调整模型权重
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals
