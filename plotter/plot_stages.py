import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from util import shapiro_wilk_test, read_csv, convert_ns_to_format, COLORS, median_largest_diff_95ci, mean_largest_diff_95ci

def add_plot_stages_subparser(parser):
    subparser = parser.add_parser('stages', help='')
    subparser.add_argument('files', type=str, nargs='+', help='The path to the csv file containing the measurements')
    subparser.add_argument('--strategy', choices=['mean', 'median'], default='median', help='The strategy used to summarize the runs')
    subparser.add_argument('--unit', choices=['s', 'ms', 'us', 'ns'], default='ms', help='The time unit, used for the histogram')
    subparser.add_argument('--absolute', action='store_true', help='shows absolute values instead precentage')
    subparser.add_argument('--out', type=str, default='plot.pdf', help='specifies the output file')
    subparser.add_argument('--skip-first-n-iterations', type=int, default=0, help='The number of first iterations to skip (used to skip warm-up iterations)')


def plot_stages(files, strategy, unit, absolute, skip_first_n_iterations, out):

    # read in data
    data_frames = [read_csv(file) for file in files]

    # pick single iterations and skip first n iterations
    data_frames = [df[df['iterations'] == 1] for df in data_frames]
    iterations = int((len(data_frames[0]) - skip_first_n_iterations * 4) / 4)
    for i in range(4):
        start = i*iterations
        end = i*iterations + skip_first_n_iterations
        data_frames = [df.drop(df.index[range(start, end)]) for df in data_frames]
        
     # Check unit
    assert np.all(np.array([df['time_unit'] == 'ns' for df in data_frames]))

    # keep unordered data
    unordered_data = np.vstack([np.vstack(df.groupby(['name']).real_time.apply(np.array).values) for df in data_frames])

    # compute medians of each stage for all scenes
    data_frames = [df.groupby(['name']).median() if strategy == 'median' else df.groupby(['name']).mean() for df in data_frames]

    # extract stage names
    permutation = get_stage_order_permutation(data_frames[0].index.to_numpy(dtype=str))
    stage_names = np.array([stage.split('/')[0].split('_')[-1] for stage in data_frames[0].index.to_numpy()])
    stage_names = stage_names[permutation]


    barplot_data = []
    scene_names = []
    for df in data_frames:
        
        # permute stages/rows in right order
        stages_full_name = df.index.to_numpy(dtype=str)
        permutation = get_stage_order_permutation(stages_full_name)
        df = df.loc[stages_full_name[permutation]]

        # get scene names
        scene_name = stages_full_name[0].split('/')[1]
        scene_names.append(scene_name)

        # get barplot data
        barplot_data.append(df['real_time'].to_numpy())

    barplot_data = convert_ns_to_format(np.array(barplot_data), unit)
    if not absolute: 
        barplot_data = barplot_data / np.sum(barplot_data, axis=1)[:, None]
    barplot_data = np.transpose(barplot_data)

    assert len(barplot_data) <= len(COLORS) 
    colors = COLORS[:len(barplot_data)]
    fig, ax = plt.subplots(figsize=(10, 7))
    barplot_data_cum_sum = np.cumsum(barplot_data,axis=0)
    for i, row in enumerate(barplot_data):
        if i == 0:
            ax.bar(scene_names, barplot_data[0], width=0.4, label=stage_names[0], color=colors[i])
        else:
            ax.bar(scene_names, barplot_data[i], width=0.4, bottom=barplot_data_cum_sum[i-1], label=stage_names[i], color=colors[i])


    ax.set_ylabel('Execution time shares' if not absolute else 'Execution time [{}]'.format(unit), fontsize=16, labelpad=10, fontweight='bold')
    #ax.set_xlabel('Scenes')
    #ax.set_title('Share of duration for all stages', fontsize=20, pad=15)
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')

    # set border
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    # set grid
    ax.yaxis.grid()
    ax.set_axisbelow(True)

    # add statistical data
    text_props = dict(facecolor='wheat', edgecolor='black', boxstyle='round,pad=0.5')
    if strategy == 'median':
        diff = median_largest_diff_95ci(unordered_data)
        ci_diff = "%.2f" % (diff*100)
        stat_str = r'95% CI: $\pm$ '
    else:
        diff = mean_largest_diff_95ci(unordered_data)
        ci_diff = "%.2f" % (diff*100)
        stat_str = r'95% CI(mean): $\pm$ '

    stat_str += ci_diff + r'%' 
    #stat_str += "\n"
    #stat_str += r'measurements: ' + str(len(unordered_data[0]))

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.925, stat_str, transform=ax.transAxes, fontsize=14, fontweight='bold', verticalalignment='top', horizontalalignment='left', bbox=text_props)

    plt.legend(bbox_to_anchor=(1.01,0.5), loc="center left")

    plt.savefig(out, bbox_inches="tight")

def get_stage_order_permutation(array):

    permutation = [
        get_index_of_maching_value(array, 'GenerateMortonCodes'),
        get_index_of_maching_value(array, 'SortMortonCodes'),
        get_index_of_maching_value(array, 'BuildTreeHierarchy'),
        get_index_of_maching_value(array, 'ComputeBoundingBoxes')
    ]
    return np.ndarray.flatten(np.array(permutation))

def get_index_of_maching_value(array, substring):
    return np.where(np.char.find(array, substring)>=0)

