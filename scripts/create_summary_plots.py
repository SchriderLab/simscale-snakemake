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
plt.rcParams.update({'font.size': 14})
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


def plot_data(data, Qs, muts, mut_labels, xlim, plot_title):
    fig, axs = plt.subplots(len(muts), 1, sharey=False, sharex=False, figsize=(6, 3*len(muts)))
    summary_df = pd.DataFrame(columns = ['Mutation', 'Q', 'Mean'])
    if len(muts) == 1:
        axs = [axs]  # Convert axs to a list if there's only one element in muts
    n_colors = len(Qs)
    colors = ListedColormap(plt.cm.viridis(np.linspace(0, 1, n_colors)))
    for i, (mut, label) in enumerate(zip(muts, mut_labels)):
        for j, Q in enumerate(Qs):
            mut_df = data[data['Q'] == Q].fillna(0)
            if plot_title == 'Fixation Times':
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
            x_plt = list(range(0, len(y) * bin_width, bin_width)) if plot_title == 'Fixation Times' else list(range(0, len(y), 1))
            bar_width = bin_width / (len(Qs) + 2) if plot_title == 'Fixation Times' else 1 / (len(Qs) + 2)
            color = colors(j)
            axs[i].bar([x + j * bar_width for x in x_plt], y, width=bar_width, label=f'Q={Q}', color=color, edgecolor='black', linewidth=0.1)
        axs[i].set_title(f'{plot_title} for {label} Mutations')
        axs[i].set_xlabel('Fixation Time' if plot_title == 'Fixation Times' else 'Frequency')
        axs[i].set_ylabel('Frequency' if plot_title == 'Fixation Times' else 'Fraction of Polymorphisms')
        if xlim:
            axs[i].set_xlim(0, xlim)

    return fig, axs, summary_df

def plot_ld(data, Qs):
    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    fig = plt.figure(figsize=(6, 3))
    #ax = fig.add_axes([0,0,1,1])
    #colors = colormaps.get_cmap('viridis', len(Qs))
    n_colors = len(Qs)
    plt.size = (6, 3)
    colors = ListedColormap(plt.cm.viridis(np.linspace(0, 1, n_colors)))
    summary_df = pd.DataFrame(columns = ['Q', 'Mean'])
    for j, Q in enumerate(Qs):
        df = data[data['Q'] == Q]
        mean_value = np.mean(df['mean_ld'])
        df = df.filter(regex=r'ld_\d+')
        # average across rows
        y = df.mean(axis=0).tolist()
        summary_df = pd.concat([summary_df, pd.DataFrame({'Q': Q, 'Mean': mean_value}, index=[0])], ignore_index=True)
        # if not np.isclose(sum(y), 0):
        #     y = [x/sum(y) for x in y]

        x_plt = list(range(1, len(y) + 1, 1))
        bar_width = 1 / (len(Qs) + 2)
        color = colors(j)
        plt.bar([x + j*bar_width for x in x_plt], y, width=bar_width, label=f'Q={Q}', color=color, edgecolor='black', linewidth=0.1)
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
    #ax = fig.add_axes([0,0,1,1])
    plt.size = (6, 3)
    summary_df = pd.DataFrame(columns = ['Mutation', 'Q', 'Mean'])
    for i, (mut, mut_label) in enumerate(zip(muts, mut_labels)):
        mut_no = int(mut[1:]) - 1
        probs = []
        for Q in Qs:
            df = data[data['Q'] == Q]
            fixation_prob = df[f'fixation_prob_{mut_no}'].tolist()
            fixation_prob = (np.mean(fixation_prob))
            probs.append(fixation_prob)
            summary_df = pd.concat([summary_df, pd.DataFrame({'Mutation': mut_label, 'Q': Q, 'Mean': fixation_prob}, index=[0])], ignore_index=True)
        bar_width = 1 / (len(muts) + 2)
        x_plt = [x + 1 for x in range(len(Qs))]
        color = mut_colors_dict[mut_label]
        plt.bar([x + i * bar_width for x in x_plt] , probs, width = bar_width, color = color,
                  label = f'{mut_label} Mutations', edgecolor='black', linewidth=0.1)
        plt.yscale('log')
        #plt.ylim(1e-6, 2e-5)
    
    plt.xticks([x + 1 for x in range(len(Qs))], Qs)
    plt.title('Fractions of Mutations Fixed')
    plt.xlabel('Q')
    plt.ylabel('Fraction of Mutations Fixed')
    return fig, summary_df
        

if plot_type == 'fixation':
    fig, axs, summary_df = plot_data(data, Qs, muts, mut_labels, xlim, 'Fixation Times')
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor = (0, 1, 1, 0.2), mode="expand", ncol = len(labels), prop={'size': 12})
    plt.savefig(output_dir / 'graphs/plot_fixation.svg', bbox_inches='tight')
    summary_df.to_csv(output_dir / 'summary_stats/plot_fixation.csv', index=False)

if plot_type == 'sfs':
    fig, axs, summary_df = plot_data(data, Qs, muts, mut_labels, xlim, 'SFS')
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor = (0, 1, 1, 0.2), mode="expand", ncol = len(labels), prop={'size': 12})
    plt.savefig(output_dir / 'graphs/plot_sfs.svg', bbox_inches='tight')
    summary_df.to_csv(output_dir / 'summary_stats/plot_sfs.csv', index=False)

if plot_type == 'ld':
    fig, summary_df = plot_ld(data, Qs)
    handles, labels = plt.gca().get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center', ncol = len(labels), bbox_to_anchor=(0.5, 1.25), prop={'size': 12})
    fig.tight_layout()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor = (0, 1.00, 1, 0.2), mode="expand", ncol = len(labels), prop={'size': 12})
    plt.savefig(output_dir / 'graphs/plot_ld.svg', bbox_inches='tight')
    summary_df.to_csv(output_dir / 'summary_stats/plot_ld.csv', index=False)

if plot_type == 'fixationprobs':
    fig, summary_df = plot_probs(data, Qs)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor = (0, 1.00, 1, 0.2), mode="expand", ncol = 2, prop={'size': 12}, columnspacing=0.5)
    #legend._legend_box.width = 100
    #fig.legend
    fig.tight_layout()
    plt.savefig(output_dir / 'graphs/plot_fixationprobs.svg', bbox_inches='tight')
    summary_df.to_csv(output_dir / 'summary_stats/plot_fixationprobs.csv', index=False)