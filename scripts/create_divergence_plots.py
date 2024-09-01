import argparse 
from scipy.stats import entropy
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np

def plot_divergence(input_file, output_dir, sample_size, muts, mut_labels, divergence_type, stat):
    mut_colors = ListedColormap(plt.cm.viridis(np.linspace(0, 1, 3)))
    mut_colors_dict = {'Neutral': mut_colors(0), 'Beneficial': mut_colors(1), 'Deleterious': mut_colors(2), 'All': '#1f77b4'}
    data = pd.read_csv(input_file)
    Qs = data['Q'].unique().tolist()
    Qs.sort()
    lowest_Q = Qs.pop(0)
    data_lowest_Q = data[data['Q'] == lowest_Q]
    plt.rcParams.update({'font.size': 14})
    div_df = pd.DataFrame(columns=['Q', 'mutation', column_dict[divergence_type],'std_dev'])
    lgnd_labels = []
    if not muts or stat == 'ld':
        muts = ['all_muts']
        mut_labels = ['All']
    for i, (mut, mut_label) in enumerate(zip(muts, mut_labels)):
        divergence_mean_mut = []
        divergence_stdev_mut = []
        divergence_lst_mut = []
        for Q in Qs:
            data_Q = data[data['Q'] == Q]
            if divergence_type == 'kl' or divergence_type == 'average_rmse':
                if stat == 'fixation':
                    data_mut_full = data_Q.filter(regex=f'{mut}_fixation') 
                    data_lowest_mut_full = data_lowest_Q.filter(regex=f'{mut}_fixation')
                elif stat == 'sfs':
                    data_mut_full = data_Q.filter(regex=f'{mut}_sfs') 
                    data_lowest_mut_full = data_lowest_Q.filter(regex=f'{mut}_sfs')
                elif stat == 'ld':
                    data_mut_full = data_Q.filter(regex=r'ld_\d+')
                    data_lowest_mut_full = data_lowest_Q.filter(regex=r'ld_\d+')
                else:
                    raise ValueError('Invalid statistic. Must be one of "fixation", "sfs", "ld", or "fixationprobs"')
            elif divergence_type == 'mean_percent_error':
                if stat == 'fixation':
                    data_mut_full = data_Q[f'mean_fixation_time_{mut}'] 
                    data_lowest_mut_full = data_lowest_Q[f'mean_fixation_time_{mut}']
                elif stat == 'sfs':
                    data_mut_full = data_Q[f'mean_sfs_{mut}']
                    data_lowest_mut_full = data_lowest_Q[f'mean_sfs_{mut}']
                elif stat == 'ld':
                    data_mut_full = data_Q['mean_ld']
                    data_lowest_mut_full = data_lowest_Q['mean_ld']
                elif stat == 'fixationprobs':
                    mut_no = int(mut[1:]) - 1
                    data_mut_full = data_Q[f'fixation_prob_{mut_no}']
                    data_lowest_mut_full = data_lowest_Q[f'fixation_prob_{mut_no}']
                else:
                    raise ValueError('Invalid statistic. Must be one of "fixation", "sfs", or "ld"')
                
            divergence_lst_q = []
            for j in range(1000):
                if sample_size:
                    data_mut = data_mut_full.sample(sample_size)
                    data_lowest_mut = data_lowest_mut_full.sample(sample_size)
                else:
                    data_mut = data_mut_full.sample(frac=1, replace=True)
                    data_lowest_mut = data_lowest_mut_full.sample(frac=1, replace=True)
                data_mut = np.mean(data_mut, axis=0)
                data_lowest_mut = np.mean(data_lowest_mut, axis=0)
                if divergence_type == 'kl':
                    data_mut = [x + int(x == 0)*np.finfo(float).eps for x in data_mut]
                    data_mut = [x / (sum(data_mut) + np.finfo(float).eps) for x in data_mut]
                    data_lowest_mut = [x + int(x == 0)*np.finfo(float).eps for x in data_lowest_mut]
                    data_lowest_mut = [x / (sum(data_lowest_mut) + np.finfo(float).eps) for x in data_lowest_mut]
                    divergence_lst_q.append(entropy(data_lowest_mut, data_mut))
                elif divergence_type == 'mean_percent_error' or divergence_type == 'median_percent_error':
                    if data_mut == 0 or data_lowest_mut == 0:
                        divergence_lst_q.append(0)
                    else:
                        divergence_lst_q.append(((data_mut - data_lowest_mut)/data_lowest_mut)* 100)
                elif divergence_type == 'average_rmse':
                    # if Q == 50:
                    #     print(data_mut, data_lowest_mut)
                    divergence_lst_q.append(np.sqrt(np.mean((data_mut - data_lowest_mut)**2)))

            divergence_mean_mut.append(np.mean(divergence_lst_q))
            divergence_stdev_mut.append(np.std(divergence_lst_q))
            divergence_lst_mut.append(divergence_lst_q)
            div_df = pd.concat([div_df, pd.DataFrame({'Q': [Q], 'mutation': [mut_label], column_dict[divergence_type]: [divergence_mean_mut[-1]], 'std_dev': [divergence_stdev_mut[-1]]})])
        
        label = f'{mut_label} Mutations'

        if not sample_size:
            bar_width = 1 / (len(muts) + 2)
            plt.bar([x + 1 + i * bar_width for x in range(len(Qs))], divergence_mean_mut, width=bar_width, label=label, color=mut_colors_dict[mut_label], edgecolor='black')
            # add error bars
            plt.errorbar([x + 1 + i * bar_width for x in range(len(Qs))], divergence_mean_mut, yerr=divergence_stdev_mut, fmt='none', ecolor='black', elinewidth=2, capsize=3)
            # set the x ticks to be the Q values
            plt.xticks([x + 1 for x in range(len(Qs))], Qs)
            # if kl_divergece values are on both sides of 0, draw a gray line at y = 0
            if any(i > 0 for i in divergence_mean_mut) and any(i < 0 for i in divergence_mean_mut):
                plt.axhline(y=0, color='gray', linestyle='--')
        else:
            violin_width = 1 / (len(muts) + 2)
            # create violin plot
            violon_arts = plt.violinplot(divergence_lst_mut, positions=[x + 1 + i * violin_width for x in range(len(Qs))], widths=violin_width-0.02, showmeans=True, showextrema=False)
            if mut_label != 'All':
                for pc in violon_arts['bodies']:
                    pc.set_facecolor(mut_colors_dict[mut_label])
                    pc.set_edgecolor('black')
                    pc.set_alpha(1)
            plt.xticks([x + 1 for x in range(len(Qs))], Qs)
            # add color legend
            lgnd_labels.append(mpatches.Patch(color=mut_colors_dict[mut_label], label=label))
        if divergence_type == 'mean_percent_error' and stat == 'sfs':
            plt.title(f'{title_dict_div[divergence_type]} of Allele Frequencies', fontsize=12)
        else:
            plt.title(f'{title_dict_div[divergence_type]} of {title_dict_stat[stat]}', fontsize=12)
        plt.xlabel('Q')
        plt.ylabel(f'{title_dict_div[divergence_type]}')
        if len(muts) > 1:
            if sample_size:
                arg_max = np.argmax(np.abs(divergence_mean_mut))
                if divergence_mean_mut[arg_max] < 0:
                    if arg_max == 0:
                        bbox_to_anchor = (1, 0)
                        loc = 'lower right'
                    else:
                        bbox_to_anchor = (0, 0)
                        loc = 'lower left'
                else:
                    if arg_max == 0:
                        bbox_to_anchor = (1, 1)
                        loc = 'upper right'
                    else:
                        bbox_to_anchor = (0, 1)
                        loc = 'upper left'
                plt.legend(handles=lgnd_labels, loc=loc, bbox_to_anchor=bbox_to_anchor, prop={'size': 10})
            else:
                plt.legend(prop={'size': 10})
        plt.savefig(output_dir / f'graphs/{column_dict[divergence_type]}_{stat}_{sample_size}.svg', bbox_inches='tight')
        div_df.to_csv(output_dir / f'summary_stats/{column_dict[divergence_type]}_{stat}_{sample_size}.csv', index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', help='Input file path', required=True)
    argparser.add_argument('--output', help='Output directory', required=True)
    argparser.add_argument('--type', help='Type of divergence to plot', required=True)
    argparser.add_argument('--muts', type=str, nargs='+', help='Mutation types to include', default=None)
    argparser.add_argument('--mut_labels', type=str, nargs='+', help='Labels for mutation types', default=None)
    argparser.add_argument('--stat', type=str, help='Statistic to prase', required=True)
    argparser.add_argument('--sample_size', type=int, help='Sample size', default=None)

    args = argparser.parse_args()
    stat = args.stat
    muts = args.muts
    mut_labels = args.mut_labels
    # Sort muts and mut_labels so that the order is always Neutral -> Beneficial -> Deleterious
    def sort_key(x):
        if x == 'Neutral':
            return 0
        elif x == 'Beneficial':
            return 1
        elif x == 'Deleterious':
            return 2
        else:
            return 3

    if muts is not None and mut_labels is not None:
        muts, mut_labels = zip(*sorted(zip(muts, mut_labels), key=lambda x: sort_key(x[1])))

    if stat != 'ld':
        if muts is None or mut_labels is None:
            argparser.error('Must provide mutation types and labels if plot statistic is not ld')

    divergence_type = args.type
    input_file = Path(args.input)
    output_dir = Path(args.output)
    sample_size = args.sample_size
    sample_label = (f'_{sample_size}')

    if divergence_type not in ['kl', 'mean_percent_error', 'median_percent_error', 'average_rmse']:
        raise ValueError('Divergence type must be one of "kl" for KL divergence or "mean_percent_error" for mean percent error')

    title_dict_div = {'kl': 'KL Divergence', 'mean_percent_error': 'Mean Percent Error', 'median_percent_error': 'Median Percent Error', 'average_rmse': 'Average RMSE'}
    title_dict_stat = {'fixation': 'Fixation Times', 'sfs': 'Site Frequency Spectra', 'ld': 'Linkage Disequilibrium', 'fixationprobs': 'Fraction of Mutations Fixed'}
    column_dict = {'kl': 'kl_divergence', 'mean_percent_error': 'mean_percent_error', 'median_percent_error': 'median_percent_error', 'average_rmse': 'average_rmse'}

    plot_divergence(input_file, output_dir, sample_size, muts, mut_labels, divergence_type, stat)