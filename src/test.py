# coding=utf-8
from __future__ import print_function
import pandas as pd
import numpy as np
from apriori import Apriori
import codecs


if __name__ == "__main__":
    a = ["a", "b", "c"]
    a = zip(a, range(len(a)))
    b = dict(a)

    if "a" in b:
        print("hhhh")
    print(b)
