# -*- coding:utf-8 -*-

from __future__ import print_function

import argparse

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import utils
# from apriori import Apriori
from utils import PreProcess

logger = utils.get_logger("clustering")


def clustering(text, n):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=1000,
                                       min_df=2, use_idf=True)
    x = tfidf_vectorizer.fit_transform(text)

    km = KMeans(n_clusters=n, init='k-means++', max_iter=100, n_init=1, verbose=True)

    logger.info("Clustering for %s class" % n)
    pred = km.fit_predict(x)
    res = pd.DataFrame([text, pred]).T
    res.columns = ["text", "class"]

    return res


# def keyword_discovery(clusters, n):
#     output = codecs.open("data/apriori_output", "w", "utf-8")
#     for i in range(n):
#         apriori = Apriori(min_sup=0.15, min_conf=0.6)
#         df = clusters[clusters["class"] == i]
#         logger.info("The %s class, size is: %s" % (i, df.shape[0]))
#         output.write("\nThe %s class: \n" % i)
#         items, rules = apriori.run_apriori(df["text"].values)
#         apriori.print_results(items, rules, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', help='path to comment data')
    parser.add_argument('-n', dest='num_cluster', default=10, help='number of cluster')

    args = parser.parse_args()

    data = PreProcess(args.input_file).corpus
    data = pd.DataFrame(data)

    words = data["comment_seg"].values
    pos = data["comment_pos"].values

    adj_list = []
    for i, w in enumerate(words):
        w_p = zip(w, pos[i])
        for item in w_p:
            if item[1] == "a":
                adj_list.append(item[0])
    pd.Series(adj_list).drop_duplicates().to_csv("data/adj", index=False, header=False, encoding="utf-8")
