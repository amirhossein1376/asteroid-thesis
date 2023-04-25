import argparse
import os
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--commondir", required=True, type=str)
parser.add_argument("--outfile", required=True, type=str)
parser.add_argument("--split", type=str, default="train")

args = parser.parse_args()

md_file = os.path.join(args.commondir,"our_data/raw", f"new_{args.split}.csv")
df = pd.read_csv(md_file)
row_list = []
for idx, row in df.iterrows():
    dict1 = {}
    dict1["utt_id"] = row["path"].split(".")[0]
    dict1["text"] = row["sentence"]
    row_list.append(dict1)

df = pd.DataFrame(row_list)
df.to_csv(args.outfile, index=False)
