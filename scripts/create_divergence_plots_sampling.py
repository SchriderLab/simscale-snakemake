import argparse 
from scipy.stats import entropy
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np



argparser = argparse.ArgumentParser()
argparser.add_argument('--input', help='Input file path', required=True)
argparser.add_argument('--output', help='Output directory', required=True)
argparser.add_argument('--type', help='Type of divergence to plot', required=True)
argparser.add_argument('--muts', type=str, nargs='+', help='Mutation types to include', default=None)
argparser.add_argument('--mut_labels', type=str, nargs='+', help='Labels for mutation types', default=None)
argparser.add_argument('--stat', type=str, help='Statistic to prase', required=True)
argparser.add_argument('--sample_size', type=int, help='Sample size', required=True)

args = argparser.parse_args()
stat = args.stat
muts = args.muts
mut_labels = args.mut_labels
sample_size = args.sample_size

if stat != 'ld':
    if muts is None or mut_labels is None:
        argparser.error('Must provide mutation types and labels if plot statistic is not ld')

divergence_type = args.type
input_file = Path(args.input)
output_dir = Path(args.output)


if divergence_type not in ['kl', 'mean_percent_error', 'median_percent_error', 'average_rmse']:
    raise ValueError('Divergence type must be one of "kl" for KL divergence or "mean_percent_error" for mean percent error')

data = pd.read_csv(input_file)
Qs = data['Q'].unique().tolist()
Qs.sort()
lowest_Q = Qs.pop(0)
data_lowest_Q = data[data['Q'] == lowest_Q]

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


mut_colors = ListedColormap(plt.cm.viridis(np.linspace(0, 1, 3)))
mut_colors_dict = {'Neutral': mut_colors(0), 'Beneficial': mut_colors(1), 'Deleterious': mut_colors(2)}

if divergence_type == 'kl':
    kl_div_df = pd.DataFrame(columns=['Q', 'mutation', 'kl_divergence'])
    if stat == 'fixation':
        for i, (mut, mut_label) in enumerate(zip(muts, mut_labels)):
            kl_divergence = []
            for Q in Qs:
                data_Q = data[data['Q'] == Q]
                data_Q = data_Q.sample(sample_size)
                data_mut = data_Q.filter(regex=f'{mut}_fixation') 
                data_mut = np.average(data_mut, axis=0)
                data_mut = [x / sum(data_mut) + np.finfo(float).eps for x in data_mut]
                #print(data_mut)
                data_lowest_mut = data_lowest_Q.filter(regex=f'{mut}_fixation')
                data_lowest_mut = data_lowest_mut.sample(sample_size)
                data_lowest_mut = np.average(data_lowest_mut, axis=0)
                data_lowest_mut = [x / sum(data_lowest_mut) + np.finfo(float).eps for x in data_lowest_mut]
                #print(data_lowest_mut[data_lowest_mut == 0])
                divergence = entropy(data_lowest_mut, data_mut)
                kl_divergence.append(divergence)
                kl_div_df = pd.concat([kl_div_df, pd.DataFrame({'Q': [Q], 'mutation': [mut_label], 'kl_divergence': [divergence]})])

            label = f'{mut_label} mutations'
            bar_width = 1 / (len(muts) + 2)
            plt.bar([x + 1 + i * bar_width for x in range(len(Qs))], kl_divergence, width=bar_width, label=label, color=mut_colors_dict[mut_label])
            # set the x ticks to be the Q values
            plt.xticks([x + 1 for x in range(len(Qs))], Qs)
        plt.title('KL divergence of Fixation Times')
        plt.xlabel('Q')
        plt.ylabel('KL divergence')
        plt.legend()
        plt.savefig(output_dir / f"graphs/sampling/kl_divergence_fixation_{sample_size}.svg" , bbox_inches='tight')
        kl_div_df.to_csv(output_dir / f"summary_stats/sampling/kl_divergence_fixation_{sample_size}.csv", index=False)

    elif stat == 'sfs':
        for i, (mut, mut_label) in enumerate(zip(muts, mut_labels)):
            kl_divergence = []
            for Q in Qs:
                data_Q = data[data['Q'] == Q]
                data_Q = data_Q.sample(sample_size)
                data_mut = data_Q.filter(regex=f'{mut}_sfs') 
                data_mut = np.average(data_mut, axis=0)
                data_mut = [x / sum(data_mut) + np.finfo(float).eps for x in data_mut]
                #print(data_mut)
                data_lowest_mut = data_lowest_Q.filter(regex=f'{mut}_sfs')
                data_lowest_mut = data_lowest_mut.sample(sample_size)
                data_lowest_mut = np.average(data_lowest_mut, axis=0)
                data_lowest_mut = [x / sum(data_lowest_mut) + np.finfo(float).eps for x in data_lowest_mut]
                #print(data_lowest_mut[data_lowest_mut == 0])
                divergence = entropy(data_lowest_mut, data_mut)
                kl_divergence.append(divergence)
            label = f'{mut_label} mutations'
            bar_width = 1 / (len(muts) + 2)
            plt.bar([x + 1 + i * bar_width for x in range(len(Qs))], kl_divergence, width=bar_width, label=label, color=mut_colors_dict[mut_label])
            # set the x ticks to be the Q values
            plt.xticks([x + 1 for x in range(len(Qs))], Qs)
            kl_div_df = pd.concat([kl_div_df, pd.DataFrame({'Q': Qs, 'mutation': [mut_label] * len(Qs), 'kl_divergence': kl_divergence})])

        plt.title('KL divergence of Site Frequency Spectra')
        plt.xlabel('Q')
        plt.ylabel('KL divergence')
        plt.legend()
        plt.savefig(output_dir / f"graphs/sampling/kl_divergence_sfs_{sample_size}.svg" , bbox_inches='tight')
        kl_div_df.to_csv(output_dir / f"summary_stats/sampling/kl_divergence_sfs_{sample_size}.csv", index=False)

    elif stat == 'ld':
        kl_divergence = []
        for Q in Qs:
            data_Q = data[data['Q'] == Q]
            data_Q = data_Q.sample(sample_size)
            data_ld = data_Q.filter(regex=r'ld_\d+')
            data_ld = np.average(data_ld, axis=0)
            data_ld = [x / sum(data_ld) + np.finfo(float).eps for x in data_ld]
            data_lowest_ld = data_lowest_Q.filter(regex=r'ld_\d+')
            data_lowest_ld = data_lowest_ld.sample(sample_size)
            data_lowest_ld = np.average(data_lowest_ld, axis=0)
            data_lowest_mut = [x / sum(data_lowest_ld) + np.finfo(float).eps for x in data_lowest_ld]
            divergence = entropy(data_lowest_ld, data_ld)
            kl_divergence.append(divergence)
            kl_div_df = pd.concat([kl_div_df, pd.DataFrame({'Q': Q, 'mutation': 'all', 'kl_divergence': divergence}, index=[0])])
        plt.bar([x + 1 for x in range(len(Qs))], kl_divergence)
        plt.xticks([x + 1 for x in range(len(Qs))], Qs)
        plt.title('KL divergence of Linkage Disequilibrium')
        plt.xlabel('Q')
        plt.ylabel('KL divergence')
        plt.savefig(output_dir / f'graphs/sampling/kl_divergence_ld_{sample_size}.svg', bbox_inches='tight')
        kl_div_df.to_csv(output_dir / f'summary_stats/sampling/kl_divergence_ld_{sample_size}.csv', index=False)

elif divergence_type == 'mean_percent_error':
    df_divergence = pd.DataFrame(columns=['Q', 'mutation', 'mean_percent_error'])
    if stat == 'fixation':
        for i, (mut, mut_label) in enumerate(zip(muts, mut_labels)):
            mean_divergence = []
            for Q in Qs:
                data_Q = data[data['Q'] == Q]
                # sample without replacement using sample size
                data_Q = data_Q.sample(sample_size)
                data_mut = data_Q[f'mean_fixation_time_{mut}'].tolist()
                data_mut = np.average(data_mut)
                data_lowest_mut = data_lowest_Q.sample(sample_size)[f'mean_fixation_time_{mut}'].tolist()
                data_lowest_mut = np.average(data_lowest_mut)
                #print(data_lowest_mut[data_lowest_mut == 0])
                divergence = ((data_mut - data_lowest_mut)/data_lowest_mut)*100
                mean_divergence.append(divergence)
            label = f'{mut_label} mutations'
            bar_width = 1 / (len(muts) + 2)
            plt.bar([x + 1 + i * bar_width for x in range(len(Qs))], mean_divergence, width=bar_width, label=label, color=mut_colors_dict[mut_label])
            # set the x ticks to be the Q values
            plt.xticks([x + 1 for x in range(len(Qs))], Qs)
            df_divergence = pd.concat([df_divergence, pd.DataFrame({'Q': Qs, 'mutation': [mut_label] * len(Qs), 'mean_percent_error': mean_divergence})])
        plt.title('Mean Percent Error of Average Fixation Times')
        plt.xlabel('Q')
        plt.ylabel('Mean Percent Error')
        plt.legend()
        plt.savefig(output_dir / f'graphs/sampling/mean_percent_error_fixation_{sample_size}.svg', bbox_inches='tight')
        df_divergence.to_csv(output_dir / f'summary_stats/sampling/mean_percent_error_fixation_{sample_size}.csv', index=False)

    elif stat == 'sfs':
        for i, (mut, mut_label) in enumerate(zip(muts, mut_labels)):
            mean_divergence = []
            for Q in Qs:
                data_Q = data[data['Q'] == Q]
                data_Q = data_Q.sample(sample_size)
                data_mut = data_Q[f'mean_sfs_{mut}'].tolist()
                data_mut = np.average(data_mut)
                data_lowest_mut = data_lowest_Q.sample(sample_size)[f'mean_sfs_{mut}'].tolist()
                data_lowest_mut = np.average(data_lowest_mut)
                #print(data_lowest_mut[data_lowest_mut == 0])
                divergence = ((data_mut - data_lowest_mut)/data_lowest_mut) * 100
                mean_divergence.append(divergence)
            label = f'{mut_label} mutations'
            bar_width = 1 / (len(muts) + 2)
            plt.bar([x + 1 + i * bar_width for x in range(len(Qs))], mean_divergence, width=bar_width, label=label, color=mut_colors_dict[mut_label])
            # set the x ticks to be the Q values
            plt.xticks([x + 1 for x in range(len(Qs))], Qs)
            df_divergence = pd.concat([df_divergence, pd.DataFrame({'Q': Qs, 'mutation': [mut_label] * len(Qs), 'mean_percent_error': mean_divergence})])
        plt.title('Mean Percent Error of Average Allele Frequencies')
        plt.xlabel('Q')
        plt.ylabel('Mean Percent Error')
        plt.legend()
        plt.savefig(output_dir / f'graphs/sampling/mean_percent_error_sfs_{sample_size}.svg', bbox_inches='tight')
        df_divergence.to_csv(output_dir / f'summary_stats/sampling/mean_percent_error_sfs_{sample_size}.csv', index=False)
    
    elif stat == 'ld':
        mean_divergence = []
        for Q in Qs:
            data_Q = data[data['Q'] == Q]
            data_Q = data_Q.sample(sample_size)
            data_ld = data_Q['mean_ld'].tolist()
            data_ld = np.average(data_ld)
            data_lowest_ld = data_lowest_Q.sample(sample_size)['mean_ld'].tolist()
            data_lowest_ld = np.average(data_lowest_ld)
            divergence = ((data_ld - data_lowest_ld)/data_lowest_ld) * 100
            mean_divergence.append(divergence)
            df_divergence = pd.concat([df_divergence, pd.DataFrame({'Q': [Q], 'mean_percent_error': [divergence]})])
        plt.bar([x + 1 for x in range(len(Qs))], mean_divergence)
        plt.xticks([x + 1 for x in range(len(Qs))], Qs)
        plt.title('Mean Percent Error of Linkage Disequilibrium')
        plt.xlabel('Q')
        plt.ylabel('Mean Percent Error')
        plt.savefig(output_dir / f'graphs/sampling/mean_percent_error_ld_{sample_size}.svg', bbox_inches='tight')
        df_divergence.to_csv(output_dir / f'summary_stats/sampling/mean_percent_error_ld_{sample_size}.csv', index=False)

    elif stat == 'fixationprobs':
        mean_divergence = []
        for i, (mut, mut_label) in enumerate(zip(muts, mut_labels)):
            mean_divergence = []
            mut_no = int(mut[1:]) - 1
            for Q in Qs:
                data_Q = data[data['Q'] == Q]
                data_Q = data_Q.sample(sample_size)
                data_mut = data_Q[f'fixation_prob_{mut_no}'].tolist()
                data_mut = np.average(data_mut)
                data_lowest_mut = data_lowest_Q.sample(sample_size)[f'fixation_prob_{mut_no}'].tolist()
                data_lowest_mut = np.average(data_lowest_mut)
                #print(data_lowest_mut[data_lowest_mut == 0])
                divergence = ((data_mut - data_lowest_mut)/data_lowest_mut) * 100
                mean_divergence.append(divergence)
                df_divergence = pd.concat([df_divergence, pd.DataFrame({'Q': [Q], 'mutation': [mut_label], 'mean_percent_error': [divergence]})])
            label = f'{mut_label} mutations'
            bar_width = 1 / (len(muts) + 2)
            plt.bar([x + 1 + i * bar_width for x in range(len(Qs))], mean_divergence, width=bar_width, label=label, color=mut_colors_dict[mut_label])
            # set the x ticks to be the Q values
            plt.xticks([x + 1 for x in range(len(Qs))], Qs)
        plt.title('Mean Percent Error of Average Fixation Probabilities')
        plt.xlabel('Q')
        plt.ylabel('Mean Percent Error')
        plt.legend()
        plt.savefig(output_dir / f'graphs/sampling/mean_percent_error_fixationprobs_{sample_size}.svg', bbox_inches='tight')
        df_divergence.to_csv(output_dir / f'summary_stats/sampling/mean_percent_error_fixationprobs_{sample_size}.csv', index=False)

elif divergence_type == 'median_percent_error':
    df_divergence = pd.DataFrame(columns=['Q', 'mutation', 'median_percent_error'])
    if stat == 'fixation':
        for i, (mut, mut_label) in enumerate(zip(muts, mut_labels)):
            median_divergence = []
            for Q in Qs:
                data_Q = data[data['Q'] == Q]
                data_mut = data_Q[f'median_fixation_time_{mut}'].tolist()
                data_mut = np.average(data_mut)
                data_lowest_mut = data_lowest_Q[f'median_fixation_time_{mut}'].tolist()
                data_lowest_mut = np.average(data_lowest_mut)
                #print(data_lowest_mut[data_lowest_mut == 0])
                divergence = (data_mut - data_lowest_mut)/data_lowest_mut
                median_divergence.append(divergence)
            label = f'{mut_label} mutations'
            bar_width = 1 / (len(muts) + 2)
            plt.bar([x + 1 + i * bar_width for x in range(len(Qs))], median_divergence, width=bar_width, label=label, color=mut_colors_dict[mut_label])
            # set the x ticks to be the Q values
            plt.xticks([x + 1 for x in range(len(Qs))], Qs)
            df_divergence = pd.concat([df_divergence, pd.DataFrame({'Q': Qs, 'mutation': [mut_label] * len(Qs), 'median_percent_error': median_divergence})])
        plt.title('Median Percent Error of Fixation Times')
        plt.xlabel('Q')
        plt.ylabel('Median Percent Error')
        plt.legend()
        plt.savefig(output_dir / 'graphs/median_percent_error_fixation.svg', bbox_inches='tight')
        df_divergence.to_csv(output_dir / 'summary_stats/median_percent_error_fixation.csv', index=False)

    elif stat == 'sfs':
        for i, (mut, mut_label) in enumerate(zip(muts, mut_labels)):
            median_divergence = []
            for Q in Qs:
                data_Q = data[data['Q'] == Q]
                data_mut = data_Q[f'median_sfs_{mut}'].tolist()
                data_mut = np.average(data_mut)
                data_lowest_mut = data_lowest_Q[f'median_sfs_{mut}'].tolist()
                data_lowest_mut = np.average(data_lowest_mut)
                #print(data_lowest_mut[data_lowest_mut == 0])
                divergence = (data_mut - data_lowest_mut)/data_lowest_mut
                median_divergence.append(divergence)
            label = f'{mut_label} mutations'
            bar_width = 1 / (len(muts) + 2)
            plt.bar([x + 1 + i * bar_width for x in range(len(Qs))], median_divergence, width=bar_width, label=label, color=mut_colors_dict[mut_label])
            # set the x ticks to be the Q values
            plt.xticks([x + 1 for x in range(len(Qs))], Qs)
            df_divergence = pd.concat([df_divergence, pd.DataFrame({'Q': Qs, 'mutation': [mut_label] * len(Qs), 'median_percent_error': median_divergence})])
        plt.title('Median Percent Error of Site Frequency Spectra')
        plt.xlabel('Q')
        plt.ylabel('Median Percent Error')
        plt.legend()
        plt.savefig(output_dir / 'graphs/median_percent_error_sfs.svg', bbox_inches='tight')
        df_divergence.to_csv(output_dir / 'summary_stats/median_percent_error_sfs.csv', index=False)
    
    elif stat == 'ld':
        median_divergence = []
        for Q in Qs:
            data_Q = data[data['Q'] == Q]
            data_ld = data_Q['median_ld'].tolist()
            data_ld = np.average(data_ld)
            data_lowest_ld = data_lowest_Q['median_ld'].tolist()
            data_lowest_ld = np.average(data_lowest_ld)
            divergence = (data_ld - data_lowest_ld)/data_lowest_ld
            median_divergence.append(divergence)
            df_divergence = pd.concat([df_divergence, pd.DataFrame({'Q': [Q], 'median_percent_error': [divergence]})])
        plt.bar([x + 1 for x in range(len(Qs))], median_divergence)
        plt.xticks([x + 1 for x in range(len(Qs))], Qs)
        plt.title('Median Percent Error of Linkage Disequilibrium')
        plt.xlabel('Q')
        plt.ylabel('Median Percent Error')
        plt.savefig(output_dir / 'graphs/median_percent_error_ld.svg', bbox_inches='tight')
        df_divergence.to_csv(output_dir / 'summary_stats/median_percent_error_ld.csv', index=False)
    

elif divergence_type == 'average_rmse':
    if stat == 'ld':
        rmse = []
        for Q in Qs:
            data_Q = data[data['Q'] == Q]
            data_Q = data_Q.sample(sample_size)
            data_ld = data_Q.filter(regex=r'ld_\d+').mean(axis = 0)
            data_lowest_ld = data_lowest_Q.filter(regex=r'ld_\d+')
            data_lowest_ld = data_lowest_ld.sample(sample_size).mean(axis = 0)
            data_lowest_ld = data_lowest_ld.to_numpy()
            # remove any columns if there are 
            rmse.append(np.sqrt(np.mean((data_ld - data_lowest_ld)**2)))
        plt.bar([x + 1 for x in range(len(Qs))], rmse)
        plt.xticks([x + 1 for x in range(len(Qs))], Qs)
        plt.title('Average RMSE of Linkage Disequilibrium')
        plt.xlabel('Q')
        plt.ylabel('Average RMSE')
        plt.savefig(output_dir / f'graphs/sampling/average_rmse_ld_{sample_size}.svg', bbox_inches='tight')
        pd.DataFrame({'Q': Qs, 'average_rmse': rmse}).to_csv(output_dir / f'summary_stats/sampling/average_rmse_ld_{sample_size}.csv', index=False)

