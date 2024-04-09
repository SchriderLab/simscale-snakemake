import argparse
import pandas as pd
from tqdm import tqdm
import sys

argparser = argparse.ArgumentParser()
argparser.add_argument('--output_file', help='Output file')
argparser.add_argument('--input_files', nargs='+', help='Input file')


args = argparser.parse_args()

output_file = args.output_file
input_files = args.input_files

dfs = []

for input_file in tqdm(input_files, file=sys.stdout, total=len(input_files)):
    df = pd.read_csv(input_file)
    dfs.append(df)

full_df = pd.concat(dfs, axis=0)
non_ld_cols = [col for col in full_df.columns if 'ld' not in col]
full_df[non_ld_cols] = full_df[non_ld_cols].fillna(0)

full_df.to_csv(output_file, index=False)