# -*- coding:utf-8 -*-

from __future__ import print_function

import logging
import re

import jieba
import jieba.posseg
import pandas as pd
import numpy as np


def get_logger(name="default"):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    return logging.getLogger(name)


class IterDocument(object):
    """
    A class for reading large file memory-friendly
    """

    def __init__(self, path, sep=None):
        """
        :param path: path to the file
        :param sep: delimiter string between fields
        """
        self.path = path
        self.sep = sep

    def __iter__(self):
        """
        :return: iteration in lines
        """
        for line in open(self.path, 'r',encoding="utf-8").readlines():
            line = line.strip()
            if line == '':
                continue
            if self.sep is not None:
                yield [item for item in line.split(self.sep) if item != ' ' and item != '']
            else:
                yield line


class TextCleaner(object):
    """
    A class for cleaning text
    """
    def __init__(self, punctuation=True, number=True, normalize=True):
        """
        Initial
        :param punctuation: whether clean punctuation
        :param number: whether clean number
        :param normalize: whether normalize token
        """
        self.punc = punctuation
        self.num = number
        self.norm = normalize

        self.punctuation = IterDocument("userdict/punctuation")

    def clean(self, text):
        """
        Clean data
        :param text: the raw string
        :return: the string after cleaning
        """
        # if self.punc:
        #     for p in self.punctuation:
        #         text = re.sub(p, "", text)

        # 只保留中文的正则表达式
        cop = re.compile("[^\u4e00-\u9fa5]")
        text = cop.sub("", text)

        return text.strip()


class Segmentor(object):
    """
    A class for segmenting text
    """
    def __init__(self, user_dict=True):
        """
        Initial
        :param user_dict: whether use user dict
        """
        self.seg = jieba
        self.seg_pos = jieba.posseg
        if user_dict:
            self.seg.load_userdict("userdict/userdict")

    def seg_token(self, text):
        """
        :param text: the raw string
        :return: a list of token
        """
        return self.seg.lcut(text)

    def seg_token_pos(self, text):
        """
        :param text: the raw string
        :return: a list of token/pos
        """
        return ["%s/%s" % (token, pos) for token, pos in self.seg_pos.lcut(text)]


class PreProcess(object):
    """
    A class for feature selecting and extracting
    """
    def __init__(self, file_path):
        """
        Initial
        :param file_path: the comment data path
        """
        self.logger = get_logger("PreProcess")
        self.logger.info("load data from %s" % file_path)

        corpus = pd.read_csv(file_path, "\t")
        # print(corpus)
        self.corpus = corpus.rename(columns={"username": "username",
                                             "comment": "content",
                                             "timestamp": "timestamp",
                                             "platform": "platform"})
        # print(self.corpus)
        self.logger.info("data size: %s" % self.corpus.shape[0])

        self.cleaner = TextCleaner()
        self.seg = Segmentor()
        self.segment()

    def segment(self):
        """
        Segment text
        """
        def seg(row):
            # print(row["content"])
            s = self.cleaner.clean(row["content"])
            # print(s)
            row["content_seg"] = self.seg.seg_token(s)
            # print(row["content_seg"])
            return row

        self.corpus = self.corpus.apply(seg, axis=1)
        

    def make_data_set(self):
        text = self.corpus["content"].values
        text_seg = np.array(map(lambda x: " ".join(x), self.corpus["content_seg"].values))

        return DataSet(text, text_seg)


class DataSet(object):
    """
    A class for organize data
    """
    def __init__(self, text, text_seg):
        """
        Initialize
        :param text: the raw text data, a numpy array as array(text1, text2, ...)
        :param text_seg: the text data after cleaning and segment, a numpy array as array("token1 token2 ...", ...)
        """
        self.text = text
        self.text_seg = text_seg
        self.data_size = len(text)

    def get_batch(self, _from, _to):
        """
        Get a batch of data
        :param _from: the position to start
        :param _to: the position to end
        :return: a subset of data set
        """
        if _from == 0 and _to >= self.data_size:
            return self
        return DataSet(self.text[_from:_to], self.text_seg[_from:_to])

    def shuffle_data(self):
        """
        Shuffle data set
        :return: random shuffled data
        """
        shuffle_arrays(self.text, self.text_seg)


def shuffle_arrays(*arrays):
    """
    In-place shuffle array
    :param arrays: raw data set
    """
    rng_state = np.random.get_state()
    for array in arrays:
        np.random.shuffle(array)
        np.random.set_state(rng_state)


if __name__ == "__main__":
    file_name = "comment_data.csv"
    pre = PreProcess(file_name)
    