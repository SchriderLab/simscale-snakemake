import torch
import allel
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import optimize

def get_ld_decay(sample_file, chr_len, n_bins=50):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = allel.read_vcf(str(sample_file), fields='*')
    g = allel.GenotypeArray(a["calldata/GT"])

    g = np.array(g)
    g = torch.tensor(g, dtype=torch.float32)
    g = torch.reshape(g, (g.shape[0], g.shape[1]*2))
    filter_lst = ~torch.all(g == 1, axis=1)
    g = g[filter_lst]
    sample_idx = np.random.choice(range(g.shape[0]), min(10000, g.shape[0]), replace=False)
    g = g[sample_idx]
    #distances = np.where(distances > 0, distances, 0)
    n = g.shape[1]
    gT = torch.transpose(g, 0, 1)
    g = g.to(device)
    gT = gT.to(device)
    pAB = (g @ gT)/n
    gn_s = torch.mean(g, dim=1)
    pA, pB = torch.meshgrid(gn_s, gn_s, indexing='xy')
    del gn_s
    del g
    del gT
    D = pAB - (pA * pB)
    D_squared = D ** 2
    del D
    r_squared = (D_squared/(pA*(1-pA)*pB*(1-pB)))
    del D_squared
    del pA
    del pB


    pos = a["variants/POS"]
    pos = pos[filter_lst]
    pos = pos[sample_idx]
    pos = torch.tensor(pos)
    pos_x, pos_y = torch.meshgrid(pos, pos, indexing='xy')
    distances = pos_x - pos_y

    ld_lst = []

    dist_bounds = zip(range(0, chr_len, chr_len//n_bins),
                      range(chr_len//n_bins, 
                            chr_len + chr_len//n_bins, 
                            chr_len//n_bins))
    
    for min_dist, max_dist in dist_bounds:
        distance_mat = torch.masked_fill(distances, (distances < min_dist) | (distances > max_dist), 0)
        distance_mat = distance_mat.to(device)
        selected_r_squared = r_squared * distance_mat
        r_squared_mean = torch.sum(selected_r_squared)/torch.sum(distance_mat)
        ld_lst.append(r_squared_mean.cpu())

    return (ld_lst)

def fit_expo_ld(sample_file, chr_len):

    def poly_func(x, a, b):
        y = a + x**b
        return(y)

    ld_lst = get_ld_decay(sample_file, chr_len)
    x = list(range(1, len(ld_lst)+1))
    x = np.array(x)
    params = optimize.curve_fit(poly_func, x, ld_lst)
    to_return = list(params[0])
    to_return.append(max(ld_lst))
    return(to_return)

def get_fixation_times(fixation_file, mutation):
    df = pd.read_csv(fixation_file, index_col=None)
    df = df[df['mutation_id'] == mutation]
    df['fixation_time'] = df['fix_gen'] - df['origin_gen']
    return df

def fit_fixation_gamma(fixation_file, mutations, Q):
    fits = []
    for mut in mutations:
        fixation_df = get_fixation_times(fixation_file, mut)
        fixation_data = fixation_df['fixation_time']
        fixation_data = fixation_data * Q
        fit = stats.gamma.fit(data=fixation_data)
        fits += [fit[0], fit[2]]
    return fits

def get_sfs(sample_file, mutation_type):
    a = allel.read_vcf(sample_file, fields=['calldata/GT', 'variants/MT'])
    g = allel.GenotypeArray(a['calldata/GT'])
    m_type = a['variants/MT']
    ac = g.count_alleles()
    ac_filtered = [ac[x] for x in range(len(ac)) if m_type[x] == mutation_type]
    dac = [x[1] for x in ac_filtered]
    if len(dac) == 0:
        return [0]
    else:
        sfs = allel.sfs(dac)
        return sfs

def fit_sfs_gamma(sample_file, mutations):
    muts = [int(x[1:]) for x in mutations]
    fits = []
    for mut in muts:
        sfs = get_sfs(sample_file, mutation_type=mut)
        if len(sfs) == 1:
            fit = [0, 0, 0]
        else:
            # we have to reconstruct the sfs since stats.gamma.fit doesn't expect a histogram
            reconstruct = []
            for ix, count in enumerate(sfs):
                lst = [ix] * count
                reconstruct += lst
            fit = stats.gamma.fit(reconstruct[1:], floc=0)
            fits += [fit[0], fit[2]]
    return fits

def get_fixation_probs(fixation_prob_file, Q):
     df = pd.read_csv(fixation_prob_file)
     vals = df.iloc[:].values[0].tolist()
     vals = [x/Q for x in vals]
     return(vals)