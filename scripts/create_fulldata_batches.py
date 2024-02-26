import argparse
import numpy as np
from utils.simparse import (
    get_sfs_features, get_ld_decay_features, get_fixation_probs, get_fixation_features
)
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import torch

argparser = argparse.ArgumentParser()
argparser.add_argument('--muts', nargs='+', help='Mutations to parse', required=True)
argparser.add_argument('--muts_fix_bin_width', type=int, nargs='+', help='Bin width for fixation times', required=True)
argparser.add_argument('--output_file', help='Output file', required=True)
argparser.add_argument('--Q', type=int, help='Scaling factor', required=True)
argparser.add_argument('--chr_len', type=int, help='Length of the simulated chromosome', required=True)
argparser.add_argument('--input_files', nargs='+', help='Input files', required=True)


args = argparser.parse_args()
output_file = args.output_file
input_files = args.input_files
fixation_files, fixation_prob_files, sample_files = np.array_split(input_files, 3)
muts = args.muts
Q = args.Q
chr_len = args.chr_len
fixation_bin_widths = {x: y for x, y in zip(muts, args.muts_fix_bin_width)}

fixation_features_dict = {x: [] for x in muts}
sfs_features_dict = {x: [] for x in muts}
fixation_prob_lst = []
ld_lst = []

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}', file=sys.stdout)
sys.stdout.flush()

for fixation_file, fixation_prob_file, sample_file in tqdm(zip(fixation_files, fixation_prob_files, sample_files), 
                                                           file=sys.stdout, total=len(fixation_files)):
    for mut in muts:
        fixation_bin_width = fixation_bin_widths[mut]
        fixation_features = get_fixation_features(Path(fixation_file), mut, Q, fixation_bin_width)
        fixation_features_dict[mut].append(fixation_features)
        sfs_features = get_sfs_features(sample_file, mut)
        sfs_features_dict[mut].append(sfs_features)
    fixation_probs = get_fixation_probs(fixation_prob_file, Q)
    fixation_prob_lst.append(fixation_probs)
    ld = get_ld_decay_features(sample_file, chr_len)
    ld_lst.append(ld)
    # flush stduout
    sys.stdout.flush()

column_names = []

# for each mutation, stack the features into a matrix
features_list = []
for mut in muts:
    # add 0 padding to make all fixation features the same length
    fixation_features = fixation_features_dict[mut]
    max_len = max([len(x) for x in fixation_features_dict[mut]])
    fixation_features = np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in fixation_features_dict[mut]])
    column_names += [f'mean_fixation_time_{mut}', f'median_fixation_time_{mut}']
    column_names += [f'{mut}_fixation_{x}' for x in range(max_len -2)]
    fixation_bin_width = fixation_bin_widths[mut]
    features_list.append(fixation_features.tolist())
    # add column for bin width
    column_names += [f'bin_width_fixation_{mut}']
    features_list.append([[fixation_bin_width] for _ in range(len(fixation_features))])

    # pad and stack sfs features
    max_len = max([len(x) for x in sfs_features_dict[mut]])
    sfs_features = np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in sfs_features_dict[mut]])
    sfs_features = np.stack(sfs_features)
    column_names += [f'mean_sfs_{mut}', f'median_sfs_{mut}']
    column_names += [f'{mut}_sfs_{x}' for x in range(len(sfs_features[0]) - 2)]
    features_list.append(sfs_features.tolist())

column_names += [f'fixation_prob_{x}' for x in range(len(fixation_prob_lst[0]))]
features_list.append(fixation_prob_lst)

column_names += ['mean_ld', 'median_ld']
column_names += [f'ld_{x}' for x in range(len(ld_lst[0]) - 2)]
features_list.append(ld_lst)

features_list = np.concatenate(features_list, axis=1)
features_df = pd.DataFrame(features_list, columns=column_names)
features_df['Q'] = Q

features_df.to_csv(output_file, index=False)




