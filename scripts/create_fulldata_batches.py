import argparse
import numpy as np
from utils.simparse import fit_fixation_gamma, fit_sfs_gamma, fit_expo_ld, get_fixation_probs

argparser = argparse.ArgumentParser()
argparser.add_argument('--muts', nargs='+', help='Mutations to parse', required=True)
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

def create_features(fixation_file, fixation_prob_file, sample_file, Q):
    fixation_features = fit_fixation_gamma(fixation_file, muts, Q)
    fixprob_features = get_fixation_probs(fixation_prob_file, Q)
    sfs_features = fit_sfs_gamma(sample_file, muts)
    ld_features = fit_expo_ld(sample_file, chr_len)
    features = (fixation_features + fixprob_features + sfs_features + ld_features + [Q])
    return(features)

features_list = []
for fixation_file, fixation_prob_file, sample_file in zip(fixation_files, fixation_prob_files, sample_files):
    features = create_features(fixation_file, fixation_prob_file, sample_file, Q)
    features_list.append(features)

features = np.array(features_list)
np.save(output_file, features)




