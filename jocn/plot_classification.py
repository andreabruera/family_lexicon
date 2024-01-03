import copy
import matplotlib
import mne
import numpy
import os
import random
import scipy

from matplotlib import colors as mcolors
from matplotlib import font_manager, pyplot
from scipy import stats

from general_utils import prepare_file, prepare_folder, return_baseline

def check_statistical_significance(args, setup_data, times):

    if not type(setup_data) == numpy.array:
        setup_data = numpy.array(setup_data)
    ### Checking for statistical significance
    random_baseline = return_baseline(args)
    ### T-test
    '''
    original_p_values = stats.ttest_1samp(setup_data, \
                         popmean=random_baseline, \
                         alternative='greater').pvalue
    '''
    lower_limit = .2
    upper_limit = .8
    lower_indices = [t_i for t_i, t in enumerate(times) if t<lower_limit]
    upper_indices = [t_i for t_i, t in enumerate(times) if t>upper_limit]

    relevant_indices = [t_i for t_i, t in enumerate(times) if (t>=lower_limit and t<=upper_limit)]
    setup_data = setup_data[:, relevant_indices]
    ### TFCE correction using 1 time-point window
    ### following Leonardelli & Fairhall 2019, checking only in the range 100-750ms
    adj = numpy.zeros((setup_data.shape[-1], setup_data.shape[-1]))
    for i in range(setup_data.shape[-1]):
        win = range(1, 3)
        for window in win:
            adj[i, max(0, i-window)] = 1
            adj[i, min(setup_data.shape[-1]-1, i+window)] = 1
    adj = scipy.sparse.coo_matrix(adj)
    corrected_p_values = mne.stats.permutation_cluster_1samp_test(
                                                 setup_data-random_baseline, 
                                                 tail=1, \
                                                 #n_permutations=4000,
                                                 #adjacency=None, \
                                                 adjacency=adj, \
                                                 threshold=dict(start=0, step=0.2))[2]

    corrected_p_values = [1. for t in lower_indices] + corrected_p_values.tolist() + [1. for t in upper_indices]
    assert len(corrected_p_values) == len(times)
    print(min(corrected_p_values))
    significance = 0.05
    significant_indices = [(i, v) for i, v in enumerate(corrected_p_values) if round(v, 2)<=significance]
    semi_significant_indices = [(i, v) for i, v in enumerate(corrected_p_values) if (round(v, 2)<=0.08 and v>0.05)]
    print('Significant indices at {}: {}'.format(significance, significant_indices))

    return significant_indices, semi_significant_indices


def read_files(args, subjects):

    data = list()
    out_path = prepare_folder(args)
    for sub in range(1, subjects+1):
        out_file, language_agnostic = prepare_file(args, sub)

        file_path = os.path.join(out_path, out_file)
        
        if not os.path.exists(file_path):
            print('missing: {}'.format(file_path))
            continue
        with open(file_path) as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
        lines = [l for l in lines if l!=['']]
        ### One line for the times, one for the scores
        assert len(lines) == 2

        ### Plotting times until t_max
        t_min = -.01
        t_max = .85
        times = [float(v) for v in lines[0]]
        times = [t for t in times if t <= t_max]

        ### Collecting subject scores
        lines = [float(v) for v in lines[1]]
        lines = [lines[t_i] for t_i, t in enumerate(times) if t>=t_min]
        times = [t for t in times if t >= t_min]

        data.append(lines)
    data = numpy.array(data, dtype=numpy.float32)

    return data, times, language_agnostic

def plot_classification(args):

    # Setting font properties

    # Using Helvetica as a font
    font_folder = '/import/cogsci/andrea/dataset/fonts/'
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

    SMALL_SIZE = 23
    MEDIUM_SIZE = 25
    BIGGER_SIZE = 27

    pyplot.rc('font', size=SMALL_SIZE)          # controls default text sizes
    pyplot.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    pyplot.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    pyplot.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    pyplot.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    pyplot.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    ### Reading the files
    ### plotting one line at a time, nice and easy
    subjects = 33
    data, times, language_agnostic = read_files(args, subjects)

    ### Setting the path

    plot_path = prepare_folder(args)
    plot_path = plot_path.replace('results', 'plots')
    os.makedirs(plot_path, exist_ok=True)


    ### Preparing a double plot
    fig, ax = pyplot.subplots(nrows=2, ncols=1, \
                              gridspec_kw={'height_ratios': [4, 1]}, \
                              figsize=(16,9), constrained_layout=True)

    ### Main plot properties

    ##### Title (usually not set)
    title = 'Classification scores for {} data\n'\
                    'type of analysis {}'.format(
                    args.data_kind, args.analysis)
    title = title.replace('_', ' ')
    #ax[0].set_title(title)

    ##### Axes
    ax[0].set_xlabel('Time', labelpad=10.0, fontweight='bold')
    if args.mapping_direction == 'correlation':
        ylabel = 'Pearson correlation'
    else:
        ylabel = 'Classification accuracy'
    ax[0].set_ylabel(ylabel, labelpad=10.0, fontweight='bold')

    #### Random baseline line
    random_baseline = return_baseline(args)
    ax[0].hlines(
                 y=random_baseline, 
                 xmin=times[0], 
                 xmax=times[-1], 
                 color='darkgrey', 
                 linestyle='dashed',
                 )

    #### Setting limits on the y axes depending on the number of classes
    if random_baseline == 0.5:
        correction = 0.02
        ymin = 0.45
        ymax = 0.7
        ax[0].set_ylim(bottom=ymin, top=ymax)

    ### Plotting when stimulus appears and disappears
    ### using both a line and text
    ax[0].vlines(
                 x=0., 
                 ymin=ymin+correction,
                 ymax=ymax-correction, 
                 color='darkgrey', 
                 linestyle='dashed',
                 )
    ax[0].text(
               x=0.012, 
               y=ymin+correction, 
               s='stimulus\nappears',
               ha='left', 
               va='bottom', 
               fontsize=23,
               )
    ### Removing all the parts surrounding the plot above

    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    ### Setting up the plot below

    ### Fake line just to equate plot 0 and 1
    ax[1].hlines(
                 y=0.15, 
                 xmin=times[0],
                 xmax=times[-1], 
                 color='darkgrey',
                 linestyle='dashed', 
                 alpha=0.,
                 )
    ### Real line
    ax[1].hlines(
                 y=0.15, 
                 xmin=times[0],
                 xmax=times[-1], 
                 color='darkgrey',
                 linestyle='dashed',
                 alpha=1, 
                 linewidth=1.,
                 )

    ### Setting limits
    ax[1].set_ylim(bottom=.5, top=.0)

    ### Removing borders
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    #ax[1].spines['left'].set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    #ax[1].get_yaxis().set_visible(False)

    ### Setting p<=0.05 label in bold 
    ax[1].set_yticks([0.15])
    ax[1].set_yticklabels(['p<=0.05'])
    labels = ax[1].get_yticklabels()
    [label.set_fontweight('bold') for label in labels]

    sig_values, semi_sig_values = check_statistical_significance(args, data, times)
    sig_indices = [k[0] for k in sig_values]
    semi_sig_indices = [k[0] for k in semi_sig_values]

    assert len(data) == subjects
    ### Building the label

    ### Starting point
    #if 'coarse' in args.analysis:
    if args.input_target_model == 'coarse_category':
        label = 'people vs places'
    if args.input_target_model == 'famous_familiar':
        label = 'famous vs familiar'
    correction = 'corrected' if args.corrected else 'uncorrected'
    label = '{} - {}'.format(label, correction)

    ### Averaging the data
    average_data = numpy.average(data, axis=0)

    ### Computing the SEM
    sem_data = stats.sem(data, axis=0)

    if args.input_target_model in ['coarse_category', 'famous_familiar']:
        if args.semantic_category_two == 'all':
            if args.semantic_category_one == 'person':
                color = 'rebeccapurple'
            elif args.semantic_category_one == 'place':
                color = 'burlywood'
            else:
                colors = [k for k in dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys()]
                color = random.sample(colors, k=1)[0]

        elif args.semantic_category_one == 'all':
            if args.semantic_category_two == 'famous':
                color = 'goldenrod'
            elif args.semantic_category_two == 'familiar':
                color = 'lightseagreen'
            else:
                colors = [k for k in dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys()]
                color = random.sample(colors, k=1)[0]
        else:
            colors = [k for k in dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys()]
            color = random.sample(colors, k=1)[0]

    ### Plotting the average
    ax[0].plot(times, average_data, linewidth=1.,
              color=color)

    ### Plotting the SEM
    ax[0].fill_between(
                       times, 
                       average_data-sem_data,
                       average_data+sem_data,
                       alpha=0.05, 
                       color=color,
                       )
    
    ### Plotting statistically significant time points
    ax[0].scatter(
                    [times[t] for t in sig_indices], 
                    [numpy.average(data, axis=0)[t] for t in sig_indices], 
                    color='white', 
                    edgecolors='black', 
                    s=20., 
                    linewidth=.5,
                    )
    ax[0].scatter(
                   [times[t] for t in semi_sig_indices], 
                   [numpy.average(data, axis=0)[t] for t in semi_sig_indices], 
                    color='white', 
                    edgecolors='black', 
                    s=20., linewidth=.5,
                    marker='*')

    ### Plotting the legend in 
    ### a separate figure below
    line_counter = 1
    step = int(len(times)/7)
    if line_counter == 1:
        p_height = .1
        x_text = times[::step][0]
        y_text = 0.4
    elif line_counter == 2:
        p_height = .2
        x_text = times[::step][4]
        y_text = 0.4
    if line_counter == 3:
        x_text = .1
        y_text = 0.1
    if line_counter == 4:
        x_text = .5
        y_text = 0.1
    ax[1].scatter([times[t] for t in sig_indices], 
               [p_height for t in sig_indices], 
                    s=60., linewidth=.5, color=color)
    ax[1].scatter(x_text, y_text, 
                  s=180., color=color, 
                  label=label, alpha=1.,
                  marker='s')
    ax[1].text(x_text+0.02, y_text, label, 
              fontsize=27., fontweight='bold', 
              ha='left', va='center')

    file_name = os.path.join(
                   plot_path,
                   '{}_{}_{}_average{}_{}.txt'.format(
                            args.input_target_model,
                            args.semantic_category_one,
                            args.semantic_category_two,
                            args.average,
                            correction,
                            )
                   )
    if not language_agnostic:
        file_name = file_name.replace(args.input_target_model, '{}_{}'.format(args.input_target_model, args.language))
    with open(file_name, 'w') as o:
        o.write('Data\tsignificant time points & FDR-corrected p-value\n')
        #for l, values in sig_container.items():
        #    o.write('{}\t'.format(l))
        o.write('{}\n\n'.format(label))
        for v in sig_values:
            o.write('{}\t{}\n'.format(times[v[0]], round(v[1], 5)))

    plot_name = file_name.replace('txt', 'jpg')
    print(plot_name)
    pyplot.savefig(plot_name, dpi=600)
    #pyplot.savefig(plot_name.replace('jpg', 'svg'), dpi=600)
    pyplot.clf()
    pyplot.close(fig)
