#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : 2020-04-12 10:10
# @Author  : zhangjingbo
# @mail    : jingbo.zhang@yooli.com
# @File    : data_process.py
# @Description: 从文件加载数据并做简单处理
# @Software: PyCharm
import re
import unicodedata


class Lang:
    def __init__(self, name, mode='char'):
        self.name = name
        self.mode = mode
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: 'EOS'}
        self.n_words = 3

    def addSentence(self, sentence):
        sentence = sentence.lower()
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLang(lang1, lang2, mode, reverse=False):
    lines = open('./{}_{}.txt'.format(lang1, lang2), mode='r', encoding='utf8').readlines()
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2, mode)
        output_lang = Lang(lang1, mode)
    else:
        input_lang = Lang(lang1, mode)
        output_lang = Lang(lang2, mode)
    return input_lang, output_lang, pairs


MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepare_Data(lang1, lang2, mode, reverse):
    input_lang, output_lang, pairs = readLang(lang1, lang2, mode, reverse)
    pairs = filterPairs(pairs)

    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
