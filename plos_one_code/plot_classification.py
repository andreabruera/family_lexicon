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

from general_utils import prepare_file, prepare_folder, read_color, return_baseline


def check_statistical_significance(args, setup_data, times):

    if not type(setup_data) == numpy.array:
        setup_data = numpy.array(setup_data)
    ### Checking for statistical significance
    random_baseline = return_baseline(args)
    ### T-test
    lower_limit = .2
    upper_limit = .8
    lower_indices = [t_i for t_i, t in enumerate(times) if t<lower_limit]
    upper_indices = [t_i for t_i, t in enumerate(times) if t>upper_limit]

    relevant_indices = [t_i for t_i, t in enumerate(times) if (t>=lower_limit and t<=upper_limit)]
    #print([t for t_i, t in enumerate(times) if (t>=lower_limit and t<=upper_limit)])
    setup_data = setup_data[:, relevant_indices]
    ### TFCE correction using 1 time-point window
    ### following Leonardelli & Fairhall 2019, checking only in the range 100-750ms
    adj = numpy.zeros((setup_data.shape[-1], setup_data.shape[-1]))
    for i in range(setup_data.shape[-1]):
        #if args.subsample == 'subsample_2' or args.data_kind != 'erp':
        #win = range(1, 2)
        win = range(1, 3)
        #if args.subsample == 'subsample_2':
        #    win = range(1, 2)
        #else:
        #    win = range(1, 3)
        for window in win:
            adj[i, max(0, i-window)] = 1
            adj[i, min(setup_data.shape[-1]-1, i+window)] = 1
    adj = scipy.sparse.coo_matrix(adj)
    tfce = mne.stats.permutation_cluster_1samp_test(
                                                 setup_data-random_baseline,
                                                 tail=1, \
                                                 adjacency=adj, \
                                                 threshold=dict(start=0, step=0.2))
    corrected_p_values = tfce[2].tolist()
    t_values = tfce[0].tolist()

    all_corrected_p_values = ['na' for t in lower_indices] + corrected_p_values + ['na' for t in upper_indices]
    all_t_values = ['na' for t in lower_indices] + t_values + ['na' for t in upper_indices]
    assert len(all_corrected_p_values) == len(times)
    assert len(all_t_values) == len(times)
    significance = 0.05
    significant_indices = [(i, v) for i, v in enumerate(all_corrected_p_values) if type(v)==float and round(v, 2)<=significance]
    semi_significant_indices = [(i, v) for i, v in enumerate(all_corrected_p_values) if type(v)==float and (round(v, 2)<0.1 and v>0.05)]
    all_ps_ts = [(v, t) for v, t in zip(all_corrected_p_values, all_t_values)]
    print('Significant indices at {}: {}'.format(significance, significant_indices))

    return significant_indices, semi_significant_indices, all_ps_ts


def read_files(args, subjects):

    data = list()
    out_path = prepare_folder(args)
    for sub in range(1, subjects+1):
        if args.across_subjects and sub != subjects:
            continue

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
        t_max = .85 if args.experiment_id == 'two' else 1.25
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
    font_folder = '../../fonts/'
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

    _, __, color = read_color(args)

    SMALL_SIZE = 23
    MEDIUM_SIZE = 25
    BIGGER_SIZE = 27
    # controls default text sizes
    pyplot.rc('font', size=SMALL_SIZE)
    # fontsize of the axes title
    pyplot.rc('axes', titlesize=SMALL_SIZE)
    # fontsize of the x and y labels
    pyplot.rc('axes', labelsize=MEDIUM_SIZE)
    # fontsize of the tick labels
    pyplot.rc('xtick', labelsize=SMALL_SIZE)
    pyplot.rc('ytick', labelsize=SMALL_SIZE)
    # fontsize of the figure title
    pyplot.rc('figure', titlesize=BIGGER_SIZE)

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
    title = 'RSA encoding scores for {} data\n'\
                    'type of analysis {}'.format(
                    args.data_kind, args.analysis)
    title = title.replace('_', ' ')

    ##### Axes
    ax[0].set_xlabel('Time', labelpad=10.0, fontweight='bold')
    ylabel = 'Spearman correlation'
    ax[0].set_ylabel(ylabel, labelpad=10.0, fontweight='bold')

    #### Random baseline line
    random_baseline = return_baseline(args)
    ax[0].hlines(y=random_baseline, xmin=times[0], \
                 xmax=times[-1], color='darkgrey', \
                 linestyle='dashed')

    if args.semantic_category_one == args.semantic_category_two:
        ymin = -.05
        ymax = .13
    else:
        ymin = -.05
        ymax = .25
    ax[0].set_ylim(bottom=ymin, top=ymax)

    ### Plotting when stimulus appears and disappears
    ### using both a line and text
    y_correction = 0.01
    ax[0].vlines(x=0., ymin=ymin+y_correction, \
                 ymax=ymax-y_correction, color='darkgrey', \
                 linestyle='dashed')
    ax[0].text(x=0.012, y=ymin+y_correction, s='stimulus\nappears', \
                ha='left', va='bottom', fontsize=23)

    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    ### Setting up the plot below

    ### Fake line just to equate plot 0 and 1
    ax[1].hlines(y=0.15, xmin=times[0], \
                 xmax=times[-1], color='darkgrey', \
                 linestyle='dashed', alpha=0.)
    ### Real line
    ax[1].hlines(y=0.15, xmin=times[0], \
                 xmax=times[-1], color='darkgrey', \
                 linestyle='dashed', alpha=1, linewidth=1.)

    ### Setting limits
    ax[1].set_ylim(bottom=.5, top=.0)

    ### Removing borders
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].get_xaxis().set_visible(False)

    ### Setting p<=0.05 label in bold
    ax[1].set_yticks([0.15])
    ax[1].set_yticklabels(['p<=0.05'])
    labels = ax[1].get_yticklabels()
    [label.set_fontweight('bold') for label in labels]

    sig_values, semi_sig_values, all_values = check_statistical_significance(args, data, times)
    sig_indices = [k[0] for k in sig_values]
    semi_sig_indices = [k[0] for k in semi_sig_values]

    assert len(data) == subjects
    label = 'RSA encoding - {}'.format(args.input_target_model)

    ### entities and correction

    correction = 'corrected' if args.corrected else 'uncorrected'
    label = '{} - {}'.format(label, correction)

    ### Averaging the data
    average_data = numpy.average(data, axis=0)

    ### Computing the SEM
    sem_data = stats.sem(data, axis=0)

    ### Plotting the average
    ax[0].plot(
               times,
               average_data,
               linewidth=2.,
               color=color,
               )

    ### Plotting the SEM
    ax[0].fill_between(
                       times,
                       average_data-sem_data,
                       average_data+sem_data,
                       alpha=0.1,
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
    ax[1].scatter(
                  [times[t] for t in sig_indices],
                  [p_height for t in sig_indices],
                  s=60.,
                  linewidth=.5,
                  edgecolors='white',
                  color=color,
                  )
    ax[1].scatter(x_text, y_text, \
                  s=180., color=color, \
                  label=label, alpha=1.,\
                  marker='s')
    ax[1].text(x_text+0.02, y_text, label, \
              fontsize=27., fontweight='bold', \
              ha='left', va='center')
    file_name = os.path.join(
                   plot_path,
                   '{}_{}_{}_{}_{}_average{}_{}.txt'.format(
                            args.input_target_model,
                            args.semantic_category_one,
                            args.semantic_category_two,
                            args.data_kind,
                            '{}ms'.format(args.temporal_resolution),
                            args.average,
                            correction,
                            )
                   )
    if not language_agnostic:
        file_name = file_name.replace(args.input_target_model, '{}_{}'.format(args.input_target_model, args.language))
    with open(file_name, 'w') as o:
        o.write('Data\ttime point\tt-value\tFDR-corrected p-value\n')
        o.write('{}\n\n'.format(label))
        assert len(times) == len(all_values)
        for time, v in zip(times, all_values):
            t = v[1]
            p = v[0] if v[0]=='na' else round(v[0], 5)
            o.write('{}\t{}\t{}\n'.format(time, t, p))

    ### Plotting
    plot_name = file_name.replace('txt', 'jpg')
    print(plot_name)
    pyplot.savefig(plot_name, dpi=600)
    pyplot.savefig(plot_name.replace('jpg', 'svg'), dpi=600)
    pyplot.clf()
    pyplot.close(fig)
