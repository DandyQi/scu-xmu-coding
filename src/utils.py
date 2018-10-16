# -*- coding:utf-8 -*-

from __future__ import print_function

import logging

import jieba
import jieba.posseg
import pandas as pd
import numpy as np


def get_logger(name="default"):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    return logging.getLogger(name)


def make_stopwords(stop_words_path=None):
    if stop_words_path is None:
        return None

    lines = list(IterDocument(stop_words_path))
    stop_words = zip(lines, range(len(lines)))
    return dict(stop_words)


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
        for line in open(self.path, 'r').readlines():
            line = line.strip().decode("utf-8")
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

        self.punctuation = IterDocument("resource/punctuation")
        self.stop_words = make_stopwords("resource/stopwords")

    def clean(self, text):
        """
        Clean data
        :param text: the raw string
        :return: the string after cleaning
        """
        if self.punc:
            for p in self.punctuation:
                text = text.replace(p.encode("utf-8"), "")

        return text.strip()

    def remove_stopwords(self, text):
        return [w for w in text if w not in self.stop_words]


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
            self.seg.load_userdict("../resource/userdict")
            self.seg.load_userdict("../resource/financial_complete.txt")
            self.seg.load_userdict("../resource/P2P_Online_loan_platform_2017.txt")
            self.seg.load_userdict("../resource/platform_name.txt")

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

        self.corpus = corpus.rename(columns={"评论ID": "id",
                                             "评论内容": "content",
                                             "平台名称": "platform",
                                             "发布时间": "timestamp",
                                             "用户名称": "user"}).drop_duplicates(subset=["id"])

        self.logger.info("data size: %s" % self.corpus.shape[0])

        self.cleaner = TextCleaner()
        self.seg = Segmentor()
        self.segment()

        print(self.corpus.head(20))

    def segment(self):
        """
        Segment text
        """
        def seg(row):
            row["content_seg"] = self.cleaner.remove_stopwords(self.seg.seg_token(self.cleaner.clean(row["content"])))
            # row["content_seg"] = self.seg.seg_token(self.cleaner.clean(row["content"]))
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

    def save_data(self):
        df = pd.DataFrame([self.text, self.text_seg]).T
        df.to_csv("data/comment_data", sep="\t", encoding="utf-8", index=None)
        logging.info("Save data to data/comment_data")

    @staticmethod
    def load_data():
        df = pd.read_csv("data/comment_data", sep="\t", encoding="utf-8", names=["text", "text_seg"], header=1).dropna()
        return DataSet(df["text"].values, df["text_seg"].values)


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
    # file_name = "data/comment"
    # pre = PreProcess(file_name)
    cleaner = TextCleaner()
    segmentor = Segmentor()

    test_text = "何事，西风。悲画扇？！你我还行"
    test_text = cleaner.clean(test_text)
    print(test_text.decode("utf-8"))
    test_text = segmentor.seg_token(test_text)
    print(" ".join(test_text))
    test_text = cleaner.remove_stopwords(test_text)
    print(" ".join(test_text))
