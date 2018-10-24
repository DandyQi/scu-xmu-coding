# coding: utf-8

import argparse
import pandas as pd
import numpy as np
import Levenshtein

import utils

logger = utils.get_logger("word_clustering")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', help='path to words')

    args = parser.parse_args()

    words = pd.read_csv(args.input_file, names=["words"]).values.reshape(1, -1)[0]
    logger.info("the size of words: %s" % len(words))

    sim_words = []
    for idx, word in enumerate(words):
        if len(word) < 2:
            continue
        for j in range(idx+1, len(words)):
            d = Levenshtein.distance(word, words[j])
            if d < 2:
                sim_words.append([word, words[j]])

    print(sim_words)
