import numpy as np
import matplotlib.pyplot as plt
import os
from util import shapiro_wilk_test, read_csv

def add_plot_stages_subparser(parser):
    subparser = parser.add_parser('stages', help='')
    subparser.add_argument('files', type=str, nargs='+', help='The path to the csv file containing the measurements')
    subparser.add_argument('--strategy', choices=['mean', 'median'], default='median', help='The strategy used to summarize the runs')
    subparser.add_argument('--absolute', action='store_true', help='shows absolute values instead precentage')
    subparser.add_argument('--out', type=str, default='plot.pdf', help='specifies the output file')
    subparser.add_argument('--skip-first-n-iterations', type=int, default=0, help='The number of first iterations to skip (used to skip warm-up iterations)')


def plot_stages(files, strategy, absolute, skip_first_n_iterations, out):

    # read in data
    data_frames = [read_csv(file) for file in files]

    # pick single iterations and skip first n iterations
    data_frames = [df[df['iterations'] == 1][skip_first_n_iterations:] for df in data_frames]

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

    if not absolute: 
        barplot_data = barplot_data / np.sum(barplot_data, axis=1)[:, None]
    barplot_data = np.transpose(barplot_data)


    fig, ax = plt.subplots()
    barplot_data_cum_sum = np.cumsum(barplot_data,axis=0)
    for i, row in enumerate(barplot_data):
        if i == 0:
            ax.bar(scene_names, barplot_data[0], 0.5, label=stage_names[0])
        else:
            ax.bar(scene_names, barplot_data[i], 0.5, barplot_data_cum_sum[i-1],label=stage_names[i])


    ax.set_ylabel('Execution time shares' if not absolute else 'Execution time (ns)')
    ax.set_xlabel('Scenes')
    ax.set_title('Share of duration for all stages')
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

