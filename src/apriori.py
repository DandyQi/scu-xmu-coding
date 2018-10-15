# coding=utf-8

from __future__ import print_function

from itertools import chain, combinations
from collections import defaultdict


def get_item_list(data_iterator):
    transaction_list = list()
    item_set = set()
    for record in data_iterator:
        record = record.split(" ")
        transaction = frozenset(record)
        transaction_list.append(transaction)
        for item in transaction:
            item_set.add(frozenset([item]))  # Generate 1-itemSets
    return item_set, transaction_list


def join_set(item_set, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set([i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length])


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


class Apriori(object):
    def __init__(self, min_sup, min_conf):
        self.min_sup = min_sup
        self.min_conf = min_conf

    def return_items_with_min_support(self, item_set, transaction_list, freq_set):

        """calculates the support for items in the itemSet and returns a subset
           of the itemSet each of whose elements satisfies the minimum support"""
        _itemSet = set()
        local_set = defaultdict(int)

        for item in item_set:
            for transaction in transaction_list:
                if item.issubset(transaction):
                    freq_set[item] += 1
                    local_set[item] += 1

        for item, count in local_set.items():
            support = float(count) / len(transaction_list)

            if support >= self.min_sup:
                _itemSet.add(item)

        return _itemSet

    def run_apriori(self, data_iter):
        """
        run the apriori algorithm. data_iter is a record iterator
        Return both:
         - items (tuple, support)
         - rules ((pretuple, posttuple), confidence)
        """
        item_set, transaction_list = get_item_list(data_iter)

        freq_set = defaultdict(int)
        large_set = dict()
        # Global dictionary which stores (key=n-itemSets,value=support)
        # which satisfy minSupport

        assocRules = dict()
        # Dictionary which stores Association Rules

        one_c_set = self.return_items_with_min_support(item_set,
                                                       transaction_list,
                                                       freq_set)

        current_l_set = one_c_set
        k = 2
        while current_l_set != set([]):
            large_set[k - 1] = current_l_set
            current_l_set = join_set(current_l_set, k)
            current_c_set = self.return_items_with_min_support(current_l_set,
                                                               transaction_list,
                                                               freq_set)
            current_l_set = current_c_set
            k = k + 1

        def get_support(item):
            """local function which Returns the support of an item"""
            return float(freq_set[item]) / len(transaction_list)

        to_ret_items = []
        for key, value in large_set.items():
            to_ret_items.extend([(tuple(item), get_support(item))
                                 for item in value])

        to_ret_rules = []
        for key, value in large_set.items()[1:]:
            for item in value:
                _subsets = map(frozenset, [x for x in subsets(item)])
                for element in _subsets:
                    remain = item.difference(element)
                    if len(remain) > 0:
                        confidence = get_support(item) / get_support(element)
                        if confidence >= self.min_conf:
                            to_ret_rules.append(((tuple(element), tuple(remain)),
                                                 confidence))
        return to_ret_items, to_ret_rules

    @staticmethod
    def print_results(items, rules, output):
        """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
        for item, support in sorted(items, key=lambda (item, support): support):
            output.write("item: %s , %.3f\n" % (" ".join(list(item)), support))
        output.write("\n------------------------ RULES:\n")
        for rule, confidence in sorted(rules, key=lambda (rule, confidence): confidence):
            pre, post = rule
            output.write("Rule: %s ==> %s , %.3f\n" % (" ".join(list(pre)), " ".join(list(post)), confidence))
