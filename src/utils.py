# -*- coding:utf-8 -*-

from __future__ import print_function

import logging

from pyltp import SentenceSplitter, Segmentor, Postagger, Parser
import pandas as pd
import numpy as np

CWS_MODEL = "model/cws.model"
POS_MODEL = "model/pos.model"
PARSER_MODEL = "model/parser.model"


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
        for line in open(self.path, 'r', encoding="utf-8").readlines():
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

        self.punctuation = IterDocument("resource/punctuation")
        # self.stop_words = make_stopwords("resource/stopwords")

    def clean(self, text):
        """
        Clean data
        :param text: the raw string
        :return: the string after cleaning
        """
        if self.punc:
            for p in self.punctuation:
                text = text.replace(p, "")

        if self.norm:
            pass

        return text.strip()

    def seg_sentence(self, text):
        for p in self.punctuation:
            text = text.replace(p, "。")

        return text.strip()

    # def remove_stopwords(self, text):
    #     return [w for w in text if w not in self.stop_words]


class WordNode:
    def __init__(self, token, pos, relation):
        self.token = token
        self.pos = pos
        self.relation = relation
        self.next = []

    def to_str(self):
        return "token: %s, pos: %s, relation: %s" % (self.token, self.pos, self.relation)

    def path(self):
        res = []
        queue = self.next
        path = [self]

        while len(queue):
            cur_node = queue.pop()

            if len(cur_node.next) == 0:
                path.append(cur_node)
                new_path = path.copy()
                res.append(new_path)
                path.pop()
            else:
                for node in cur_node.next:
                    queue.insert(0, node)
                path.append(cur_node)

        return res


def find_x(s, x):
    return [idx for idx, item in enumerate(s) if item[0] == x]


class SentenceParser(object):
    """
    A class for segmenting text
    """
    def __init__(self):
        """
        Initial
        """
        self.sen_split = SentenceSplitter()
        self.seg = Segmentor()
        self.seg.load_with_lexicon(CWS_MODEL, "resource/lexicon")
        self.pos = Postagger()
        self.pos.load_with_lexicon(POS_MODEL, "resource/lexicon")
        self.parser = Parser()
        self.parser.load(PARSER_MODEL)

        self.rule = IterDocument("resource/rule")

    def seg_sentence(self, text):
        return self.sen_split.split(text)

    def seg_token(self, text):
        """
        :param text: the raw string
        :return: a list of token
        """
        return self.seg.segment(text)

    def pos_tag(self, words):
        """
        :param words: the list of token
        :return: a list of pos
        """
        return self.pos.postag(words)

    def parse(self, words, pos):
        arcs = self.parser.parse(words, pos)

        # print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
        nodes = list(map(lambda x: (x.head, x.relation), arcs))

        root_idx = find_x(nodes, 0)
        root = WordNode(words[root_idx[0]], pos[root_idx[0]], nodes[root_idx[0]][1])
        tree = {root_idx[0]: root}
        queue = root_idx

        while len(queue):
            next_idx = queue.pop()
            for idx in find_x(nodes, next_idx + 1):
                queue.insert(0, idx)
                new_node = WordNode(words[idx], pos[idx], nodes[idx][1])
                tree[next_idx].next.append(new_node)
                tree[idx] = new_node

        return root

    def extract(self, path):
        res = []
        rule = self.rule
        for p in path:
            for r in rule:
                window_size = len(r.split(";"))
                if len(p) == window_size:
                    if ";".join(map(lambda x: "%s,%s" % (x.relation, x.pos), p)) == r:
                        res.append("".join(map(lambda x: x.token, p)))
                else:
                    for i in range(len(p) - window_size):
                        p_slice = ";".join(map(lambda x: "%s,%s" % (x.relation, x.pos), p[i:i+window_size]))
                        if p_slice == r:
                            res.append("".join(map(lambda x: x.token, p[i:i+window_size])))
                            break
        return res


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

        self.corpus = pd.read_csv(file_path, "\t")

        self.logger.info("data size: %s" % self.corpus.shape[0])

        self.cleaner = TextCleaner()
        self.parser = SentenceParser()
        self.segment()

    def segment(self):
        """
        Segment text
        """
        def seg_pos(row):
            words = self.parser.seg_token(self.cleaner.clean(row["comment"]))
            pos = self.parser.pos_tag(words)

            row["comment_seg"] = words
            row["comment_pos"] = pos

            return row

        self.corpus = self.corpus.apply(seg_pos, axis=1)

    def make_data_set(self):
        text = self.corpus["comment"].values
        text_seg = np.array(map(lambda x: " ".join(x), self.corpus["comment_seg"].values))

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
    cleaner = TextCleaner()
    parser = SentenceParser()

    # test_text = "一个多小时就到账了，不错不错。大平台值得推荐"
    # 资金回款很准时，提现也很快，很稳的平台，等有活动继续加仓
    test_text = "体验非常好，到期一天后到账，提现当天到，头部平台值得信任"
    clean_text = cleaner.seg_sentence(test_text)
    aspect = []
    for sen in parser.seg_sentence(clean_text):
        seg_text = parser.seg_token(cleaner.clean(sen))
        print(" ".join(seg_text))
        pos_text = parser.pos_tag(seg_text)
        print(" ".join(pos_text))
        root = parser.parse(seg_text, pos_text)
        all_path = root.path()

        for idx, p in enumerate(all_path):
            print("path %s: %s" % (idx, "\t".join(map(lambda x: x.to_str(), p))))

        aspect.extend(parser.extract(all_path))

    print(";".join(set(aspect)))
