import torch
import allel
import numpy as np
import pandas as pd
import numpy as np
from pathlib import Path

# def get_ld_decay(sample_file, chr_len, n_bins=50):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     a = allel.read_vcf(str(sample_file), fields='*')
#     g = allel.GenotypeArray(a["calldata/GT"])

#     g = np.array(g)
#     g = torch.tensor(g, dtype=torch.float32)
#     g = torch.reshape(g, (g.shape[0], g.shape[1]*2))
#     filter_lst = ~torch.all(g == 1, axis=1)
#     g = g[filter_lst]
#     sample_idx = np.random.choice(range(g.shape[0]), min(2000, g.shape[0]), replace=False)
#     g = g[sample_idx]
#     #distances = np.where(distances > 0, distances, 0)
#     n = g.shape[1]
#     gT = torch.transpose(g, 0, 1)
#     g = g.to(device)
#     gT = gT.to(device)
#     pAB = (g @ gT)/n
#     gn_s = torch.mean(g, dim=1)
#     pA, pB = torch.meshgrid(gn_s, gn_s, indexing='xy')
#     del gn_s
#     del g
#     del gT
#     D = pAB - (pA * pB)
#     D_squared = D ** 2
#     del D
#     r_squared = (D_squared/(pA*(1-pA)*pB*(1-pB)))
#     del D_squared
#     del pA
#     del pB


#     pos = a["variants/POS"]
#     pos = pos[filter_lst]
#     pos = pos[sample_idx]
#     pos = torch.tensor(pos)
#     pos_x, pos_y = torch.meshgrid(pos, pos, indexing='xy')
#     distances = pos_x - pos_y

#     ld_lst = []

#     dist_bounds = zip(range(0, chr_len, chr_len//n_bins),
#                       range(chr_len//n_bins, 
#                             chr_len + chr_len//n_bins, 
#                             chr_len//n_bins))
    
#     for min_dist, max_dist in dist_bounds:
#         distance_mat = torch.masked_fill(distances, (distances < min_dist) | (distances > max_dist), 0)
#         distance_mat = distance_mat.to(device)
#         selected_r_squared = r_squared * distance_mat
#         r_squared_mean = (torch.sum(selected_r_squared)/torch.sum(distance_mat)).cpu().item()
        
#         ld_lst.append(r_squared_mean)

#     return (ld_lst)

def get_ld_decay(sample_file, chr_len, n_bins=50):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    a = allel.read_vcf(str(sample_file), fields='*')

    g = allel.GenotypeArray(a["calldata/GT"])
    g = np.array(g)
    g = torch.tensor(g, dtype=torch.float32)
    g = torch.reshape(g, (g.shape[0], g.shape[1]*2))
    filter_lst = ~torch.all(g == 1, dim=1)
    g = g[filter_lst]
    
    sample_idx = np.random.choice(range(g.shape[0]), min(5000, g.shape[0]), replace=False)
    sample_idx = np.sort(sample_idx)
    g = g[sample_idx]
    # #distances = np.where(distances > 0, distances, 0)
    n = g.shape[1]
    g = g.to(device)
    gT = torch.transpose(g, 0, 1)
    pAB = (g @ gT)/n
    pAb = (g @ (1-gT))/n
    paB = ((1-g) @ gT)/n
    pab = ((1-g) @ (1-gT))/n

    gn_s = torch.mean(g, dim=1)
    pA, pB = torch.meshgrid(gn_s, gn_s, indexing='xy')

    del gn_s
    del g
    del gT
    D = pAB * (pab) - (pAb * paB)

    D_squared = D**2
    del D

    r_squared = (D_squared/(pA*(1-pA)*pB*(1-pB)))
    r_squared = torch.tril(r_squared, diagonal=-1)

    del D_squared
    del pA
    del pB

    tril_indices = torch.tril_indices(r_squared.shape[0], r_squared.shape[1], -1)
    r_squared_vals = r_squared[tril_indices[0], tril_indices[1]]
    r_squared_mean = torch.mean(r_squared_vals)
    r_squared_median = torch.median(r_squared_vals)
    pos = a["variants/POS"]
    pos = pos[filter_lst]
    pos = pos[sample_idx]
    pos = torch.tensor(pos)

    pos_x, pos_y = torch.meshgrid(pos, pos, indexing='xy')
    distances = pos_y - pos_x

    distances.to(device)

    ld_lst = []


    distance_masks = torch.div(distances, (chr_len//n_bins), rounding_mode='floor') + 1
    distance_masks = torch.tril(distance_masks, diagonal=-1)


    for i in range(1 , n_bins + 1):
        dist_mask = distance_masks == i

        selected_r_squared = r_squared[dist_mask]
        selected_mean = torch.mean(selected_r_squared).cpu().item()
        
        ld_lst.append(selected_mean)

    return r_squared_mean.cpu().item(), r_squared_median.cpu().item(), ld_lst 

def get_ld_decay_features(sample_file, chr_len, n_bins=50):
    r_squared_mean, r_squared_median, ld_lst  = get_ld_decay(sample_file, chr_len, n_bins)
    return [r_squared_mean, r_squared_median] + ld_lst

def get_fixation_times(fixation_file: Path, mutation: str, Q: int) -> np.array:
    df = pd.read_csv(fixation_file, index_col=None)
    df = df[df['mutation_id'] == mutation]
    if len(df) == 0:
        return np.array([0])
    fixation_times = ((df['fix_gen'] - df['origin_gen'])*Q).to_numpy()
    return fixation_times
     
def get_fixation_features(fixation_file: Path, mutation: str, Q: int, bin_width = 'NA'):
    fixation_times = get_fixation_times(fixation_file, mutation, Q)
    fixation_mean = np.mean(fixation_times)
    fixation_median = np.median(fixation_times)
    if bin_width is not None:
        bins = np.arange(0, max(fixation_times) + bin_width, bin_width)
    else:
        bins = 'auto'

    hist, _ = np.histogram(fixation_times, bins=bins)

    return [fixation_mean, fixation_median] + hist.tolist()

def get_sfs(sample_file, mutation_type):
    a = allel.read_vcf(sample_file, fields=['calldata/GT', 'variants/MT'])
    g = allel.GenotypeArray(a['calldata/GT'])
    m_type = a['variants/MT']
    ac = g.count_alleles()
    mutation_type = int(mutation_type[1:])
    ac_filtered = [ac[x] for x in range(len(ac)) if m_type[x] == mutation_type]
    dac = [x[1] for x in ac_filtered]
    if len(dac) == 0:
        return [0]
    else:
        sfs = allel.sfs(dac)
        return sfs

def get_sfs_features(sample_file, mutation_type):
    sfs = get_sfs(sample_file, mutation_type)
    reconstructed_sfs = []
    for i in range(len(sfs)):
        reconstructed_sfs += [i] * sfs[i]
    mean = np.mean(reconstructed_sfs)
    median = np.median(reconstructed_sfs)
    return [mean, median] + list(sfs)

def get_fixation_probs(fixation_prob_file, Q):
     df = pd.read_csv(fixation_prob_file)
     vals = df.iloc[:].values[0].tolist()
     vals = [x/Q for x in vals]
     return(vals)