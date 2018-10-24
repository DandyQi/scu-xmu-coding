# -*- coding:utf-8 -*-

import argparse
import time
import pandas as pd
from openpyxl import load_workbook


def normalize_time_content(row):
    t = row["timestamp"]
    if len(t) == 13:
        t = t[0:10]
    else:
        t = int(time.mktime(time.strptime(t, "%Y/%m/%d")))
    row["timestamp"] = t
    row["comment"] = row["comment"].replace("\n", "").strip()
    return row


if __name__ == "__main__":
    df = pd.read_csv("../data/source_data.csv", names=["username", "comment", "timestamp", "platform"])
    df = df.apply(normalize_time_content, axis=1)
    df = df.drop_duplicates()

    print(df.shape)

    df.to_csv("../data/comment_data.csv", sep="\t", index=False, encoding="utf-8")

    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('input_file', help='path to xlsx file')
    # parser.add_argument('output_file', help='path to output file')
    # args = parser.parse_args()
    #
    # wb = load_workbook(filename=args.input_file)
    # sheets = wb.get_sheet_names()
    # ws = wb.get_sheet_by_name(sheets[0])
    # rows = ws.rows
    # columns = ws.columns
    #
    # content = []
    # for row in rows:
    #     line = [col.value for col in row]
    #     content.append(line)
    #
    # df = pd.DataFrame(content)
    # df.to_csv(args.output_file, sep='\t', index=False, header=False, encoding='utf-8')
