# coding: utf-8

import argparse
import pandas as pd


import utils

logger = utils.get_logger("word_clustering")
cleaner = utils.TextCleaner()
parser = utils.SentenceParser()


def extract(row):
    text = row["comment"]
    clean_text = cleaner.split_sentence(text)
    aspect = []
    all_path = []
    for sen in parser.seg_sentence(clean_text):
        seg_text = parser.seg_token(cleaner.clean(sen))
        pos_text = parser.pos_tag(seg_text)
        root = parser.parse(seg_text, pos_text)
        path = root.path()
        all_path.extend(path)
        aspect.extend(parser.extract(path))

    row["path"] = "||".join(map(lambda p: ";".join(map(lambda x: x.to_str(), p)), all_path))
    row["aspect"] = ";".join(set(aspect))

    return row


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description=__doc__)
    argParser.add_argument('input_file', help='path to words')

    args = argParser.parse_args()

    comment = pd.read_csv(args.input_file, sep="\t")
    logger.info("the size of comment: %s" % comment.shape[0])

    comment = comment.apply(extract, axis=1)

    comment.to_csv("data/aspect", index=False, header=True, encoding="utf-8", sep="\t")
