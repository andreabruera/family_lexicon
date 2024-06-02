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

from general_utils import ColorblindPalette, colors_mapper, prepare_file, prepare_folder, read_colors, return_baseline
#from plotting.plot_violins import plot_violins


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
    #lower_limit = 0.2 if args.experiment_id == 'two' else 0.1
    lower_limit = .2
    #upper_limit = 1.
    #upper_limit = 0.8 if args.experiment_id == 'two' else 1.
    upper_limit = .8 if args.experiment_id == 'two' else 1.2
    #lower_limit = 0.3 if args.experiment_id == 'two' else 0.3
    #upper_limit = 1.2 if args.experiment_id == 'two' else 1.2
    lower_indices = [t_i for t_i, t in enumerate(times) if t<lower_limit]
    upper_indices = [t_i for t_i, t in enumerate(times) if t>upper_limit]

    relevant_indices = [t_i for t_i, t in enumerate(times) if (t>=lower_limit and t<=upper_limit)]
    setup_data = setup_data[:, relevant_indices]
    if args.data_kind not in [
                              'erp', 
                              'D26',
                              'B16',
                              'C22',
                              'C23',
                              'C11',
                              'alpha', 
                              'alpha_theta', 
                              'beta', 
                              'lower_gamma', 
                              'higher_gamma', 
                              'delta',
                              'theta_phase'
                              'theta',
                              ]:
        ### Wilcoxon + FDR correction
        significance_data = setup_data.T - random_baseline
        original_p_values = list()
        for t in significance_data:
            p = stats.wilcoxon(t, alternative='greater')[1]
            original_p_values.append(p)

        assert len(original_p_values) == setup_data.shape[-1]
        #corrected_p_values = mne.stats.fdr_correction(original_p_values[2:6])[1]
        #corrected_p_values = original_p_values[:2] +corrected_p_values.tolist() + original_p_values[6:]
        corrected_p_values = mne.stats.fdr_correction(original_p_values)[1]
    else:
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
        #t_min = -.05
        t_min = -.01
        t_max = .85 if args.experiment_id == 'two' else 1.25
        #t_max = 1.25
        #if args.experiment_id == 'one':
        #    t_max = 1.2
        #else:
        #    t_max = 0.85
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

    colors = ColorblindPalette()
    if args.semantic_category_one == 'all' and args.semantic_category_two == 'all': 
        alt = True
    else:
        alt = False
    if args.input_target_model == 'xlm-roberta-large_individuals':
        model_key = 0
    elif args.input_target_model == 'w2v_sentence_individuals':
        model_key = 1
    elif args.input_target_model == 'perceptual_individuals':
        model_key = 2
    elif args.input_target_model == 'affective_individuals':
        model_key = 0
        alt = True
    else:
        model_key = random.randint(0, 2)
        alt = random.randint(0, 1)
        alt = True if alt == 1 else False

    colors_dict = read_colors()
    colors_mapper_dict = colors_mapper()

    color_key = (args.semantic_category_one, args.semantic_category_two)
    if color_key in colors_mapper_dict.keys():
        color = colors_dict[model_key][colors_mapper_dict[color_key]]
    else:
        random_colors = [k for k in dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys()]
        color = random.sample(random_colors, k=1)[0]

    # Setting font properties

    # Using Helvetica as a font
    font_folder = '/import/cogsci/andrea/dataset/fonts/'
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

    wong_palette = ['goldenrod', 'skyblue', \
                    'mediumseagreen', 'chocolate', \
                    'palevioletred', 'khaki']

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
    #plot_path = os.path.join('plots', 
    #                         args.experiment_id, 
    #                         args.data_kind, 
    #                         args.analysis,
    #                         args.entities,
    #                         args.semantic_category,
    #                         )
    plot_path = plot_path.replace('results', 'plots')
    os.makedirs(plot_path, exist_ok=True)


    ### Preparing a double plot
    fig, ax = pyplot.subplots(nrows=2, ncols=1, \
                              gridspec_kw={'height_ratios': [4, 1]}, \
                              figsize=(16,9), constrained_layout=True)
    #fig.tight_layout()

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
    ax[0].hlines(y=random_baseline, xmin=times[0], \
                 xmax=times[-1], color='darkgrey', \
                 linestyle='dashed')

    #### Setting limits on the y axes depending on the number of classes
    if random_baseline == 0.25:
        ymin = 0.15
        ymax = 0.35
        correction = 0.01
        ax[0].set_ylim(bottom=ymin, top=ymax)
    else:
        if random_baseline == 0.:
            correction = 0.01
            #ymin = -.1
            #ymax = .15
            #ymax = .25
            if args.evaluation_method == 'r_squared':
                ymin = -5
                ymax = -1
            else:
                ymin = -.05
                ymax = .15
            ax[0].set_ylim(bottom=ymin, top=ymax)
        if random_baseline == 0.5:
            correction = 0.02
            ymin = 0.45
            ymax = 0.7
            #ymax = 0.61
            ax[0].set_ylim(bottom=ymin, top=ymax)
        elif random_baseline == 0.125:
            correction = 0.01
            ymin = 0.075
            ymax = 0.175
            ax[0].set_ylim(bottom=ymin, top=ymax)

    if args.evaluation_method == 'correlation':
        if args.experiment_id == 'one':
            if args.semantic_category_one == args.semantic_category_two:
                ymin = -.05
                ymax = .06
            else:
                ymin = -.05
                ymax = .11
        else:
            if args.semantic_category_one == args.semantic_category_two:
                ymin = -.05
                ymax = .13
            else:
                ymin = -.05
                ymax = .25
        ax[0].set_ylim(bottom=ymin, top=ymax)

    ### Plotting when stimulus appears and disappears
    ### using both a line and text
    ax[0].vlines(x=0., ymin=ymin+correction, \
                 ymax=ymax-correction, color='darkgrey', \
                 linestyle='dashed')
    ax[0].text(x=0.012, y=ymin+correction, s='stimulus\nappears', \
                ha='left', va='bottom', fontsize=23)
    #stim_disappears = 0.75 if args.experiment_id=='one' else 0.5
    #ax[0].vlines(x=stim_disappears, ymin=ymin+correction, \
    #             ymax=ymax-correction, color='darkgrey', \
    #             linestyle='dashed')
    #ax[0].text(x=stim_disappears+0.02, y=ymin+correction, s='stimulus\ndisappears', \
    #            ha='left', va='bottom', fontsize=23)
    ### Removing all the parts surrounding the plot above

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
    #ax[1].spines['left'].set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    #ax[1].get_yaxis().set_visible(False)

    ### Setting p<=0.05 label in bold 
    ax[1].set_yticks([0.15])
    ax[1].set_yticklabels(['p<=0.05'])
    labels = ax[1].get_yticklabels()
    [label.set_fontweight('bold') for label in labels]

    ### Setting colors for plotting different setups
    #cmap = cm.get_cmap('plasma')
    #colors = [cmap(i) for i in range(32, 220, int(220/len(folders)))]

    #number_lines = len([1 for k, v in data_dict.items() \
    #                   for d in v.keys()])

    #line_counter = 0
    #sig_container = dict()

    #import pdb; pdb.set_trace()
    #for cat, values in v.items():

    #    for e_type, subs in values.items():

    #        #if args.semantic_category in ['people', 'places']:
    #        #    color = colors[cat]
    #        #else:
    #        #if args.semantic_category == 'people':
    #        if cat == 'people':
    #            color = 'steelblue'
    #        #elif args.semantic_category == 'places':
    #        elif cat == 'places':
    #            color = 'darkorange'
    #        #elif args.semantic_category == 'familiar':
    #        elif cat == 'familiar':
    #            color = 'mediumspringgreen'
    #        #elif args.semantic_category == 'famous':
    #        elif cat == 'famous':
    #            color = 'magenta'
    #        else:
    #            #color = colors[e_type]
    #            #color = 'goldenrod'

    ### Plot is randomly colored
    #color = (numpy.random.random(), numpy.random.random(), numpy.random.random())

    sig_values, semi_sig_values = check_statistical_significance(args, data, times)
    sig_indices = [k[0] for k in sig_values]
    semi_sig_indices = [k[0] for k in semi_sig_values]

    assert len(data) == subjects
    '''
    if args.experiment_id == 'one':
        label = '{}{}'.format(cat, e_type).replace(\
                                           '_', ' ')
    elif args.experiment_id == 'two':
        if args.analysis == 'classification_coarse':
            split_label = 'people vs places'
        elif args.analysis == 'classification_famous_familiar':
            split_label = 'famous vs familiar'
        if args.corrected:
            split_label = '{} - corrected for length'.format(split_label)
        else:
            split_label = '{} - uncorrected'.format(split_label)
        label = '{} - N={}'.format(split_label, len(subs))
        if args.semantic_category in ['people', 'places',
                                'famous', 'familiar']:
            label =  label.replace(split_label, '{}, {}'.format(split_label, args.semantic_category))                    
    '''
    ### Building the label

    ### Starting point
    #if 'coarse' in args.analysis:
    if args.input_target_model == 'coarse_category':
        label = 'people vs places'
    if args.input_target_model == 'fine_category':
        label = 'fine-grained classification'
    if args.input_target_model == 'famous_familiar':
        label = 'famous vs familiar'
    else:
        label = 'RSA - {}'.format(args.input_target_model)

    ### entities and correction
    
    #if args.analysis in ['time_resolved_rsa' 'time_resolved_rsa_encoding']:
    #    correction = args.input_target_model
    #else:
    correction = 'corrected' if args.corrected else 'uncorrected'
    label = '{} - {}'.format(label, correction)

    #if 'whole_trial' in args.analysis:
    #    continue_marker = False
    #    title = 'Accuracy on {}\n{}'.format(args.analysis.\
    #                                  replace('_', ' '), \
    #                                  args.entities)
    #    accs = numpy.array([[v[0] for v in subs]])
    #    labels = [e_type]
    #    
    #    plot_violins(accs, labels, plot_path, \
    #                 title, random_baseline)
    #
    #else:
    #    continue_marker = True

    ### Averaging the data
    average_data = numpy.average(data, axis=0)

    ### Computing the SEM
    sem_data = stats.sem(data, axis=0)

    #if args.subsample == 'subsample_2':
    #    ### Interpolate averages
    #    inter = scipy.interpolate.interp1d(times, \
    #                    average_data, kind='cubic')
    #    x_plot = numpy.linspace(min(times), max(times), \
    #                            num=500, endpoint=True)
    #    #ax[0].plot(times, average_data, linewidth=.5)
    #    ax[0].plot(x_plot, inter(x_plot), linewidth=1., \
    #              color=color)
    #else:

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
    #ax[0].errorbar(x=times, y=numpy.average(subs, axis=0), \
                   #yerr=stats.sem(subs, axis=0), \
                   #color=colors[f_i], \
                   #elinewidth=.5, linewidth=1.)
                   #linewidth=.5)

    ### Plotting the SEM
    ax[0].fill_between(times, average_data-sem_data, \
                       average_data+sem_data, \
                       alpha=0.05, color=color)
    #for t_i, t in enumerate(times):
        #ax[0].violinplot(dataset=setup_data[:, t_i], positions=[t_i], showmedians=True)
    
    ### Plotting statistically significant time points
    ax[0].scatter([times[t] for t in sig_indices], \
    #ax[0].scatter(significant_indices, \
               [numpy.average(data, axis=0)[t] \
                    for t in sig_indices], \
                    color='white', \
                    edgecolors='black', \
                    s=20., linewidth=.5)
    ax[0].scatter([times[t] for t in semi_sig_indices], \
    #ax[0].scatter(significant_indices, \
               [numpy.average(data, axis=0)[t] \
                    for t in semi_sig_indices], \
                    color='white', \
                    edgecolors='black', \
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
    '''
    if line_counter <= number_lines/3:
        x_text = .1
        y_text = 0.+line_counter*.1
    elif line_counter <= number_lines/3:
        x_text = .4
        y_text = 0.+line_counter*.066
    elif line_counter > number_lines/3:
        x_text = .7
        y_text = 0.+line_counter*.033
    '''

    ax[1].scatter([times[t] for t in sig_indices], \
    #ax[0].scatter(significant_indices, \
               [p_height for t in sig_indices], \
                    s=60., linewidth=.5, color=color)
    ax[1].scatter(x_text, y_text, \
                  #color=colors[f_i], \
                  s=180., color=color, \
                  label=label, alpha=1.,\
                  marker='s')
    ax[1].text(x_text+0.02, y_text, label, \
              fontsize=27., fontweight='bold', \
              ha='left', va='center')

    '''
    for sub in subs:
        ax[0].plot(times, sub, alpha=0.1)
    '''
    #sig_container[label] = sig_values

    #if continue_marker:

    ### Writing to file the significant points
    #if 'searchlight' in args.analysis:
    #    file_name = os.path.join(plot_path,\
    #                   'cluster_{}_{}_{}_{}_{}_{}_{}.txt'.format(electrode, cat, args.analysis, e_type,\
    #                    args.data_kind, hz, args.subsample))
    #else:
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
        o.write('Data\tsignificant time points & FDR-corrected p-value\n')
        #for l, values in sig_container.items():
        #    o.write('{}\t'.format(l))
        o.write('{}\n\n'.format(label))
        for v in sig_values:
            o.write('{}\t{}\n'.format(times[v[0]], round(v[1], 5)))

    ### Plotting
    #if 'searchlight' in args.analysis:
    #    plot_name = os.path.join(plot_path,\
    #                   #'{}_{}_{}_{}_{}_{}.pdf'.format(cat, args.analysis, e_type, \
    #                   'cluster_{}_{}_{}_{}_{}_{}_{}.jpg'.format(electrode, cat, args.analysis, e_type, \
    #                    args.data_kind, hz, args.subsample))
    #else:
    #    plot_name = os.path.join(plot_path,\
    #                   #'{}_{}_{}_{}_{}_{}.pdf'.format(cat, args.analysis, e_type, \
    #                   '{}_{}_{}_{}_{}_{}.jpg'.format(cat, args.analysis, e_type, \
    #                    args.data_kind, hz, args.subsample))
    plot_name = file_name.replace('txt', 'jpg')
    print(plot_name)
    pyplot.savefig(plot_name, dpi=600)
    pyplot.savefig(plot_name.replace('jpg', 'svg'), dpi=600)
    pyplot.clf()
    pyplot.close(fig)
