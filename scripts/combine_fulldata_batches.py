import argparse
import pandas as pd
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument('--muts', nargs='+', help='Mutations to parse')
argparser.add_argument('--output_file', help='Output file')
argparser.add_argument('--input_files', nargs='+', help='Input file')


args = argparser.parse_args()

output_file = args.output_file
input_files = args.input_files
muts = args.muts

column_names = []
column_names += [f'{x}_{y}' for x in muts for y in ['fix_shape', 'fix_scale']]
column_names += [f'{x}_{y}' for x in muts for y in ['fixprob']]
column_names += [f'{x}_{y}' for x in muts for y in ['sfs_shape', 'sfs_scale']]
column_names += ['ld_a', 'ld_b', 'ld_max', 'Q']

full_df_lst = []


for input_file in input_files:
    df = np.load(input_file)
    df = pd.DataFrame(df)
    full_df_lst.append(df)

full_df = pd.concat(full_df_lst, axis=0)
full_df.columns = column_names
full_df.to_csv(output_file, index=False)