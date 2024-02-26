import pandas as pd 
import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument('--input', help='Input file path', required=True)
argparser.add_argument('--output', help='Output directory', required=True)
argparser.add_argument('--type', help='Type of plot to create', required=True)
argparser.add_argument('--muts', type=str, nargs='+', help='Mutation types to include', default=None)
argparser.add_argument('--mut_labels', type=str, nargs = '+', help='Labels for mutation types', default=None)
argparser.add_argument('--xlim', type=int, help='X-axis limits', default=None)

args = argparser.parse_args()

plot_type = args.type
muts = args.muts
mut_labels = args.mut_labels
if plot_type != 'ld':
    if muts is None or mut_labels is None:
        argparser.error('Must provide mutation types and labels if plot type is not ld')

input_file = Path(args.input)
output_dir = Path(args.output)
plot_type = args.type

xlim = args.xlim

data = pd.read_csv(input_file)
Qs = data['Q'].unique().tolist()
Qs.sort()

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'


def plot_data(data, Qs, muts, mut_labels, xlim, plot_title):
    fig, axs = plt.subplots(len(muts), 1, sharey=False, sharex=False, figsize=(6, 9))
    summary_df = pd.DataFrame(columns = ['Mutation', 'Q', 'Mean'])
    if len(muts) == 1:
        axs = [axs]  # Convert axs to a list if there's only one element in muts
    n_colors = len(Qs)
    colors = ListedColormap(plt.cm.viridis(np.linspace(0, 1, n_colors)))
    for i, (mut, label) in enumerate(zip(muts, mut_labels)):
        for j, Q in enumerate(Qs):
            mut_df = data[data['Q'] == Q]
            if plot_title == 'Fixation times':
                bin_width = int(mut_df[f'bin_width_fixation_{mut}'].tolist()[0])
                mean_value = np.mean(mut_df[f'mean_fixation_time_{mut}'].tolist())
                mut_df = mut_df.filter(regex=f'{mut}_fixation')
            elif plot_title == 'SFS':
                bin_width = 1
                mean_value = np.mean(mut_df[f'mean_sfs_{mut}'].tolist())
                mut_df = mut_df.filter(regex=f'{mut}_sfs')
            # average across rows
            y = mut_df.mean(axis=0).tolist()

            summary_df = pd.concat([summary_df, pd.DataFrame({'Mutation': label, 'Q': Q, 'Mean': mean_value}, index=[0])], ignore_index=True)
            if not np.isclose(sum(y), 0):
                y = [x / sum(y) for x in y]
            x_plt = list(range(0, len(y) * bin_width, bin_width)) if plot_title == 'Fixation times' else list(range(0, len(y), 1))
            bar_width = bin_width / (len(Qs) + 2) if plot_title == 'Fixation times' else 1 / (len(Qs) + 2)
            color = colors(j)
            axs[i].bar([x + j * bar_width for x in x_plt], y, width=bar_width, label=f'Q={Q}', color=color)
        axs[i].set_title(f'{plot_title} for {label.lower()} mutations')
        axs[i].set_xlabel('Fixation time' if plot_title == 'Fixation times' else 'Frequency')
        axs[i].set_ylabel('Frequency')
        if xlim:
            axs[i].set_xlim(0, xlim)

    return fig, axs, summary_df

def plot_ld(data, Qs):
    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    fig = plt.figure(figsize=(6, 3))
    #colors = colormaps.get_cmap('viridis', len(Qs))
    n_colors = len(Qs)
    colors = ListedColormap(plt.cm.viridis(np.linspace(0, 1, n_colors)))
    summary_df = pd.DataFrame(columns = ['Q', 'Mean'])
    for j, Q in enumerate(Qs):
        df = data[data['Q'] == Q]
        mean_value = np.mean(df['mean_ld'])
        df = df.filter(regex=r'ld_\d+')
        # average across rows
        y = df.mean(axis=0).tolist()
        summary_df = pd.concat([summary_df, pd.DataFrame({'Q': Q, 'Mean': mean_value}, index=[0])], ignore_index=True)
        if not np.isclose(sum(y), 0):
            y = [x/sum(y) for x in y]

        x_plt = list(range(0, len(y), 1))
        bar_width = 1 / (len(Qs) + 2)
        color = colors(j)
        plt.bar([x + j*bar_width for x in x_plt], y, width=bar_width, label=f'Q={Q}', color=color)
    plt.title(f'Linkage Disequilibrium')
    # set x-axis ticks to be at every 1 
    #axs[i].set_xticks([x for x in range(0, len(y), 1)])
    # rotate x-axis ticks 90 degrees
    #axs[i].tick_params(axis='x', rotation=90)
    plt.xlabel('Genomic Bin')
    plt.ylabel('Frequency')

    return fig, summary_df

def plot_probs(data, Qs):
    fig = plt.figure(figsize=(6, 3))

    n_colors = len(muts)
    colors = ListedColormap(plt.cm.viridis(np.linspace(0, 1, n_colors)))
    summary_df = pd.DataFrame(columns = ['Mutation', 'Q', 'Mean'])
    for i, (mut, mut_label) in enumerate(zip(muts, mut_labels)):
        mut_no = int(mut[1:]) - 1
        probs = []
        for Q in Qs:
            df = data[data['Q'] == Q]
            fixation_prob = df[f'fixation_prob_{mut_no}'].tolist()
            fixation_prob = np.mean(fixation_prob)
            probs.append(fixation_prob)
            summary_df = pd.concat([summary_df, pd.DataFrame({'Mutation': mut_label, 'Q': Q, 'Mean': fixation_prob}, index=[0])], ignore_index=True)
        bar_width = 1 / (len(muts) + 2)
        x_plt = [x + 1 for x in range(len(Qs))]
        plt.bar([x + i * bar_width for x in x_plt] , probs, width = bar_width, color = colors(i),
                  label = f'{mut_label} mutations')
    
    plt.xticks([x + 1 for x in range(len(Qs))], Qs)
    plt.title('Fixation probabilities')
    plt.xlabel('Q')
    plt.ylabel('Fixation probability')
    return fig, summary_df
        

if plot_type == 'fixation':
    fig, axs, summary_df = plot_data(data, Qs, muts, mut_labels, xlim, 'Fixation times')
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.tight_layout()
    #fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.06,0.96))
    fig.legend(handles, labels, loc='upper center', ncol = len(labels), bbox_to_anchor=(0.5, 1.06))
    #fig.savefig(output_file, dpi=800, bbox_inches='tight')
    plt.savefig(output_dir / 'graphs/fixation.svg', bbox_inches='tight')
    summary_df.to_csv(output_dir / 'summary_stats/fixation.csv', index=False)

if plot_type == 'sfs':
    fig, axs, summary_df = plot_data(data, Qs, muts, mut_labels, xlim, 'SFS')
    handles, labels = plt.gca().get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.08,0.96))
    fig.tight_layout()
    fig.legend(handles, labels, loc='upper center', ncol = len(labels), bbox_to_anchor=(0.5, 1.06))
    #fig.legend(handles, labels, loc='outside right upper')

    plt.savefig(output_dir / 'graphs/sfs.svg', bbox_inches='tight')
    summary_df.to_csv(output_dir / 'summary_stats/sfs.csv', index=False)

if plot_type == 'ld':
    fig, summary_df = plot_ld(data, Qs)
    handles, labels = plt.gca().get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center', ncol = len(labels), bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    plt.savefig(output_dir / 'graphs/ld.svg', bbox_inches='tight')
    summary_df.to_csv(output_dir / 'summary_stats/ld.csv', index=False)

if plot_type == 'fixation_probs':
    fig, summary_df = plot_probs(data, Qs)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol = len(labels), bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    plt.savefig(output_dir / 'graphs/fixation_probs.svg', bbox_inches='tight')
    summary_df.to_csv(output_dir / 'summary_stats/fixation_probs.csv', index=False)