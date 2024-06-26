import argparse
import mne
import numpy
import os
import re
import scipy
import sklearn

from matplotlib import pyplot
from mne import stats

from scipy import spatial, stats
from skbold.preproc import ConfoundRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import SVR
from tqdm import tqdm

from io_utils import ExperimentInfo, LoadEEG

def return_baseline(args):

    random_baseline = 0.5

    return random_baseline

def read_args():
    semantic_categories_one = [
                         ### both categories
                         'all',
                         ### only one alternation
                         'person', 
                         'place', 
                         ]
    semantic_categories_two = [
                         'all',
                         ### experiment two
                         'famous', 
                         'familiar',
                         ]

    parser = argparse.ArgumentParser()

    ### plotting
    parser.add_argument(
                        '--plot', 
                        action='store_true', 
                        default=False, 
                        help='Plotting instead of testing?'
                        )
    ### applying to anything
    parser.add_argument(
                        '--average', 
                        type=int, choices=list(range(25)), 
                        default=24, 
                        help='How many ERPs to average?'
                        )
    parser.add_argument('--analysis', 
                        type=str,
                        choices=[
                                 'time_resolved',
                                 'searchlight',
                                 ],
                        required=True,
                        )
    parser.add_argument('--mapping_model', 
                        type=str,
                        choices=[
                                 'ridge',
                                 'support_vector',
                                 'rsa',
                                 ],
                        default='rsa', 
                        )
    ### Arguments which do not affect output folder structure
    parser.add_argument(
                        '--data_folder',
                        type=str,
                        required=True, 
                        help='Indicates where the experiment files are stored',
                        )
    parser.add_argument(
                        '--debugging',
                        action='store_true',
                        default=False, 
                        help='No multiprocessing')

    parser.add_argument(
                        '--semantic_category_one', 
                        type=str,
                        choices=semantic_categories_one
                        )
    parser.add_argument(
                        '--semantic_category_two', 
                        type=str,
                        choices=semantic_categories_two,
                        )
    parser.add_argument(
                        '--corrected',
                        action='store_true',
                        default=False, 
                        help='Controlling test samples for length?')
    parser.add_argument(
                        '--comparison',
                        action='store_true',
                        )
    ### Enc-decoding specific
    parser.add_argument(
                        '--input_target_model',
                        required=True,
                        choices=[
                                 ### Category
                                 'coarse_category',
                                 'famous_familiar',
                                 # orthography
                                 'orthography',
                                 'word_length',
                                 ],
                        help='Which computational model to use for decoding?'
                        )
    parser.add_argument(
                        '--cores_usage',
                        choices=[
                                 'max',
                                 'high',
                                 'mid',
                                 'low',
                                 'min'
                                 ],
                        default = 'mid',
                        )
    args = parser.parse_args()

    check_args(args)
    
    return args

def check_args(args):
    ### checking inconsistencies in the args
    marker = False
    if args.input_target_model == 'coarse_category' and args.semantic_category_one in ['person', 'place']:
        marker = True
        message = 'wrong model and semantic category!'
    if args.input_target_model == 'famous_familiar' and args.semantic_category_two in ['famous', 'familiar']:
        marker = True
        message = 'wrong model and semantic category!'
    if marker:
        raise RuntimeError(message)

def split_train_test(args, split, eeg, experiment, comp_vectors):

    ### Selecting the relevant index 
    ### for the trigger_to_info dictionary
    if args.input_target_model == 'coarse_category':
        cat_index = 1
    if args.input_target_model == 'famous_familiar':
        cat_index = 2

    test_model = list()
    test_brain = list()
    test_lengths = list()
    test_trigs = list()

    for trig in split:
        trig = experiment.trigger_to_info[trig][0]
        vecs = comp_vectors[trig]
        erps = eeg[trig]
        if not isinstance(erps, list):
            erps = [erps]
            vecs = [vecs]
        test_model.extend(vecs)
        test_brain.extend(erps)
        test_lengths.append(len(trig))
        test_trigs.append(trig)

    train_model = list()
    train_brain = list()
    train_trigs = list()
    train_lengths = list()

    for k, erps in eeg.items():
        k = {v[0] : k for k, v in experiment.trigger_to_info.items()}[k]
        if not isinstance(erps, list):
            erps = [erps]
        if k not in split:
            k = experiment.trigger_to_info[k][0]
            vecs = [comp_vectors[k]]
            train_model.extend(vecs)
            train_brain.extend(erps)
            train_lengths.append(len(k))
            train_trigs.append(k)

    test_model = numpy.array(test_model, dtype=numpy.float64)
    train_model = numpy.array(train_model, dtype=numpy.float64)

    train_brain = numpy.array(train_brain, dtype=numpy.float64)
    test_brain = numpy.array(test_brain, dtype=numpy.float64)

    assert train_model.shape[0] == len(eeg.keys())-2
    assert len(test_model) == 2

    train_lengths = numpy.array(train_lengths)
    test_lengths = numpy.array(test_lengths)

    assert train_lengths.shape[0] == train_model.shape[0]
    assert test_lengths.shape[0] == test_model.shape[0]

    return train_brain, test_brain, train_model, test_model, train_lengths, test_lengths, train_trigs, test_trigs

def correct_for_length(args, train_brain, train_lengths, test_brain, test_lengths):
    '''
    ### only train
    cfr = ConfoundRegressor(confound=train_lengths, X=train_true.copy())
    cfr.fit(train_true)
    train_true = cfr.transform(train_true)
    '''
    ### train + test
    correction_lengths = numpy.hstack([train_lengths, test_lengths])
    correction_data = numpy.vstack([train_brain, test_brain])
    cfr = ConfoundRegressor(confound=correction_lengths, X=correction_data)
    cfr.fit(train_brain)
    train_brain = cfr.transform(train_brain)
    test_brain = cfr.transform(test_brain)

    return train_brain, test_brain

def remove_average(train_brain, test_brain):
    ### train + test
    #correction_data = numpy.average(numpy.vstack([train_brain, test_brain]), axis=0)
    ### train
    correction_data = numpy.average(train_brain, axis=0)
    train_brain = [tr - correction_data for tr in train_brain]
    test_brain = [tst - correction_data for tst in test_brain]

    return train_brain, test_brain

def evaluate_pairwise(args, train_brain, test_brain, train_model, test_model, train_lengths, test_lengths):

    if args.mapping_model == 'rsa':

        test_input = [[scipy.stats.pearsonr(tst, tr)[0] for tr in train_brain] for tst in test_brain]
        test_target = test_model.copy()

    elif args.mapping_model in ['ridge', 'support_vector']:

        ### Differentiating between binary and multiclass classifier
        if args.mapping_model == 'support_vector':
            classifier = SVR()
        elif args.mapping_model == 'ridge':
            classifier = Ridge()
            #print('loading ridge')
            #classifier = RidgeCV(alphas=(0.00005, 1, 10))
        train_model = scipy.stats.zscore(train_model)
        test_model = scipy.stats.zscore(test_model)
        train_brain = scipy.stats.zscore(train_brain, axis=1)
        test_brain = scipy.stats.zscore(test_brain, axis=1)

        classifier.fit(train_brain, train_model)
        test_target = test_model.copy()
        test_input = classifier.predict(test_brain)

    wrong = 0.
    for idx_one, idx_two in [(0, 1), (1, 0)]:
        ### decoding
        if type(test_target[0]) in [int, float, numpy.float64]:
            wrong += 1 - abs(test_input[idx_one]-test_target[idx_two])
        else:
            wrong += scipy.stats.pearsonr(test_input[idx_one], test_target[idx_two])[0]

    correct = 0.
    for idx_one, idx_two in [(0, 0), (1, 1)]:
        ### decoding
        if type(test_target[0]) in [int, float, numpy.float64]:
            correct += 1 - abs(test_input[idx_one]-test_target[idx_two])
        else:
            correct += scipy.stats.pearsonr(test_input[idx_one], test_target[idx_two])[0]

    if correct > wrong:
        #accuracies.append(1)
        accuracy = 1
    else:
        #accuracies.append(0)
        accuracy = 0

    return accuracy

def evaluation_round(args, experiment, current_eeg, comp_vectors):

    scores = list()

    for split in experiment.test_splits:
        train_brain, test_brain, train_model, test_model, train_lengths, test_lengths, train_trigs, test_trigs = split_train_test(args, split, current_eeg, experiment, comp_vectors)

        ### if label's the same, stop!
        if type(test_model[0]) in [int, float, numpy.float64, tuple]:
            if test_model[0] == test_model[1]:
                continue

        ### regress out word_length
        if args.corrected:
            train_brain, test_brain = correct_for_length(args, train_brain, train_lengths, test_brain, test_lengths)
        else:
            train_brain, test_brain = remove_average(train_brain, test_brain)

        ### transforming model to pairwise if required
        if args.mapping_model == 'rsa':
            if type(test_model[0]) in [int, float, numpy.float64]:
                test_model = [[1 - abs(tst-tr) for tr in train_model] for tst in test_model]
            ### for vectors, we use correlation
            ### NB: the higher the value, the more similar the items!
            else:
                test_model = [[scipy.stats.pearsonr(tst, tr)[0] for tr in train_model] for tst in test_model]
            #test_model = [model_sims[tuple(sorted([tr, tst]))] for tr in train_trigs for tst in test_trigs]

        ### rsa decoding
        score = evaluate_pairwise(args, train_brain, test_brain, train_model, test_model, train_lengths, test_lengths)
        scores.append(score)

    corr = numpy.average(scores)

    return corr

def prepare_folder(args):

    out_path = os.path.join(
                            'results', 
                            args.analysis,
                            args.mapping_model,
                            'average_{}'.format(args.average),
                            args.semantic_category_one,
                            args.semantic_category_two,
                             )
    os.makedirs(out_path, exist_ok=True)

    return out_path

def prepare_file(args, n):

    correction = 'corrected' if args.corrected else 'uncorrected'
    file_path = 'sub_{:02}_{}_{}.scores'.format(n, correction, args.input_target_model)
    language_agnostic = True

    return file_path, language_agnostic

def how_many_cores(args):
    ### how many cores to use?
    if args.cores_usage == 'max':
        div = 1
    if args.cores_usage == 'high':
        div = 0.75
    elif args.cores_usage == 'mid':
        div = 0.5
    elif args.cores_usage == 'low':
        div = 0.25
    elif args.cores_usage == 'min':
        div = 0.1
    processes = int(os.cpu_count()*div)
    print('using {} processes'.format(processes))
    return processes

def plot_erps(args):
    if args.semantic_category_one == 'all' and args.semantic_category_two == 'all':
        elec_mapper = ['A{}'.format(i) for i in range(1, 33)] +\
                      ['B{}'.format(i) for i in range(1, 33)] +\
                      ['C{}'.format(i) for i in range(1, 33)] +\
                      ['D{}'.format(i) for i in range(1, 33)]
        elec_mapper = {e_i : e for e_i,e in enumerate(elec_mapper)}
        inverse_mapper = {v : k for k, v in elec_mapper.items()}
        ### read zones
        zones = {i : list() for i in range(1, 14)}
        with open(os.path.join('data', 'ChanPos.tsv')) as i:
            counter = 0
            for l in i:
                if counter == 0:
                    counter += 1
                    continue
                line = l.strip().split('\t')
                zones[int(line[6])].append(inverse_mapper[line[0]])
        zones[14] = list(range(128))
        zone_names = {
                      #1 : 'left_frontal',
                      #2 : 'right_frontal',
                      #3 : 'left_fronto-central',
                      #4 : 'right_fronto-central',
                      #5 : 'left_posterior-central',
                      #6 : 'right_posterior-central',
                      #7 : 'left_posterior',
                      #8 : 'right_posterior',
                      #9 : 'frontal_midline',
                      #10 : 'central',
                      #11 : 'posterior_midline',
                      #12 : 'left_midline',
                      #13 : 'right_midline',
                      14 : 'whole_brain'
                      }
        output_folder = os.path.join('plots', 'erps')
        if not os.path.exists(output_folder):
            ### plotting erps...
            print('plotting erps...')
            os.makedirs(output_folder, exist_ok=True)
            erp_data = dict()
            #for across_n in range(1, 3+1):
            for across_n in range(1, 33+1):
                ### Loading the experiment
                experiment = ExperimentInfo(args, subject=across_n)
                ### Loading the EEG data
                eeg = LoadEEG(args, experiment, across_n)
                deconfounded_data = eeg.data_dict

                familiarity_mapper = {int(k) : v for k, v in zip(experiment.events_log['value'], experiment.events_log['familiarity'])}
                category_mapper = {int(k) : v for k, v in zip(experiment.events_log['value'], experiment.events_log['semantic_domain'])}
                for k, v in deconfounded_data.items():
                    key = '{}_{}'.format(category_mapper[k], familiarity_mapper[k])
                    try:
                        erp_data[key].append(v)
                    except KeyError:
                        erp_data[key] = [v]
            colors = {
                      'person famous' : 'green',
                      'person familiar' : 'black',
                      'place famous' : 'purple',
                      'place familiar' : 'red',
                      'person' : 'rebeccapurple',
                      'place' : 'burlywood',
                      'famous' : 'goldenrod',
                      'familiar' : 'lightseagreen',
                      }

            for cats_one in ['person_place', 'famous_familiar']:
                for k in cats_one.split('_'):
                    for cats_two in ['person_place', 'famous_familiar']:
                        keys = list()
                        data = dict()
                        if cats_one == cats_two:
                            if k in ['place', 'familiar']:
                                continue
                            for k_two in cats_two.split('_'):
                                rel_keys = [k for k in erp_data.keys() if k_two in k]
                                keys.append(rel_keys)
                            new_k = cats_one
                        else:
                            for k_two in cats_two.split('_'):
                                if 'person' in cats_one:
                                    key = '_'.join([k, k_two])
                                else:
                                    key = '_'.join([k_two, k])
                                keys.append(key)
                                data[key] = erp_data[key]
                            assert len(keys) == 2
                            new_k = k

                        for zone_n, zone_label in zone_names.items():
                            zone_elecs = zones[zone_n]
                            current_erp_data = {k : numpy.average(numpy.array(v)[:, zone_elecs, :], axis=1) for k, v in erp_data.items()}
                            #for k, v in erp_data.items():
                            output_file = os.path.join(
                                                output_folder, 
                                                'erp_{}_{}.jpg'.format(
                                                                          new_k,
                                                                          zone_label,
                                                                          )
                                                )
                            fig, ax = pyplot.subplots(
                                      figsize=(16,9), 
                                      constrained_layout=True,
                                      )
                            for key_i, key in enumerate(keys):
                                if type(key) == list:
                                    v = numpy.average([current_erp_data[single_k] for single_k in key], axis=0)
                                    label = new_k.split('_')[key_i]
                                else:
                                    v = current_erp_data[key]
                                    label = key.replace('_', ' ')
                                ax.plot(
                                        eeg.times, 
                                        numpy.average(v, axis=0), 
                                        color=colors[label],
                                        #label=key.replace('_', ' '),
                                        label=label
                                        )
                                ax.fill_between(
                                                x=eeg.times,
                                                y1=numpy.average(v, axis=0)-scipy.stats.sem(v, axis=0), 
                                                y2=numpy.average(v, axis=0)+scipy.stats.sem(v, axis=0), 
                                                color=colors[label],
                                                alpha=0.1,
                                                )
                            ### statistical significance
                            lower_limit = .2
                            upper_limit = .8
                            lower_indices = [t_i for t_i, t in enumerate(eeg.times) if t<lower_limit]
                            upper_indices = [t_i for t_i, t in enumerate(eeg.times) if t>upper_limit]

                            relevant_indices = [t_i for t_i, t in enumerate(eeg.times) if (t>=lower_limit and t<=upper_limit)]
                            if type(key) == list:
                                setup_data = [numpy.average([current_erp_data[single_k] for single_k in key], axis=0) for key in keys]
                            else:
                                setup_data = [current_erp_data[k] for k in keys]
                            setup_data = numpy.subtract(setup_data[0], setup_data[1])[:, relevant_indices]
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
                                                                         setup_data, 
                                                                         #tail=1,
                                                                         #n_permutations=4000,
                                                                         #adjacency=None, \
                                                                         adjacency=adj, \
                                                                         threshold=dict(start=0, step=0.2))[2]

                            corrected_p_values = [1. for t in lower_indices] + corrected_p_values.tolist() + [1. for t in upper_indices]
                            assert len(corrected_p_values) == len(eeg.times)

                            with open(output_file.replace('jpg', 'txt'), 'w') as o:
                                o.write('time point\tFDR-corrected p-value\n')
                                for v in enumerate(corrected_p_values):
                                    o.write('{}\t{}\n'.format(eeg.times[v[0]], round(v[1], 5)))

                            significance = 0.05
                            #significance = 0.1
                            significant_indices = [i for i, v in enumerate(corrected_p_values) if round(v, 2)<=significance]
                            ax.scatter(
                                     [eeg.times[t] for t in significant_indices],
                                     [-.025 for t in significant_indices],
                                      s=60., 
                                      linewidth=.5, 
                                      color='silver',
                                      )
                            #semi_significant_indices = [(i, v) for i, v in enumerate(corrected_p_values) if (round(v, 2)<=0.08 and v>0.05)]

                            ax.vlines(x=0., 
                                         ymin=-.02,
                                         ymax=.02, 
                                         color='darkgrey',
                                         linestyle='dashed')
                            ax.hlines(y=0., 
                                         xmin=min(eeg.times),
                                         xmax=max(eeg.times), 
                                         color='darkgrey',
                                         linestyle='dashed')

                            ax.set_xlim(left=-.05, right=.85)
                            ax.set_ylim(bottom=-.026, top=.026)
                            ax.legend(fontsize=23)
                            pyplot.savefig(output_file)
                            #pyplot.savefig(output_file.replace('.jpg', '.svg'))
                            pyplot.clf()
                            pyplot.close()
                            print(output_file)

def plot_scalp_erps(args, clusters):
    if args.semantic_category_one == 'all' and args.semantic_category_two == 'all':
        elec_mapper = ['A{}'.format(i) for i in range(1, 33)] +['B{}'.format(i) for i in range(1, 33)] +['C{}'.format(i) for i in range(1, 33)] +['D{}'.format(i) for i in range(1, 33)]
        output_folder = os.path.join('plots', 'scalp_erps')
        if not os.path.exists(output_folder):
            print('plotting scalp erps...')
            os.makedirs(output_folder, exist_ok=True)
            mne_adj_matrix = clusters.mne_adjacency_matrix
            print('now plotting data...')
            erp_data = dict()
            #for across_n in range(1, 3+1):
            for across_n in range(1, 33+1):
                sub_erp_data = dict()
                ### Loading the experiment
                experiment = ExperimentInfo(args, subject=across_n)
                ### Loading the EEG data
                eeg = LoadEEG(args, experiment, across_n)
                time_splits = [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8)]
                time_windows = [[t_i for t_i, t in enumerate(eeg.times) if t>=t_min and t<t_max] for t_min, t_max in time_splits]
                ### removing word length as a confound
                all_sub_data = [(k, v) for k, v in eeg.data_dict.items()]
                all_lengths = [(k, len(experiment.trigger_to_info[k][0])) for k, v in all_sub_data]
                ###
                deconfounded_data = dict()
                for k, v in all_sub_data:
                    new_v = list()
                    for t in range(v.shape[-1]):
                        t_sub_data = [(k, v[:, t]) for k, v in all_sub_data]
                        train_input = [e[:, t] for e_k, e in all_sub_data if e_k!=k]
                        #train_target = [e for e_k, e in all_lengths if e_k!=k]
                        cfr = ConfoundRegressor(
                                               confound=numpy.array([v[1] for v in all_lengths]), 
                                               X=numpy.array([v[1] for v in t_sub_data]),
                                               )
                        cfr.fit(numpy.array(train_input))
                        ### because of a bug prediction should input 2 vecz
                        pred = cfr.transform(numpy.array([v[:, t], v[:,t]]))[0]
                        new_v.append(pred)
                    deconfounded_data[k] = numpy.array(new_v).T
                    assert deconfounded_data[k].shape == v.shape

                familiarity_mapper = {int(k) : v for k, v in zip(experiment.events_log['value'], experiment.events_log['familiarity'])}
                category_mapper = {int(k) : v for k, v in zip(experiment.events_log['value'], experiment.events_log['semantic_domain'])}
                #for k, v in eeg.data_dict.items():
                for k, v in deconfounded_data.items():
                    key = '{}_{}'.format(category_mapper[k], familiarity_mapper[k])
                    try:
                        sub_erp_data[key].append(v)
                    except KeyError:
                        sub_erp_data[key] = [v]
                    #print(erp_data[key].shape)
                assert len(sub_erp_data.keys()) == 4
                for k, v in sub_erp_data.items():
                    try:
                        erp_data[k].append(numpy.average(v, axis=0))
                    except KeyError:
                        erp_data[k] = [numpy.average(v, axis=0)]
                assert len(erp_data.keys()) == 4
            erp_data = {k : numpy.array(v) for k, v in erp_data.items()}

            for cats_one in ['person_place', 'famous_familiar']:
                for k in cats_one.split('_'):
                    for cats_two in ['person_place', 'famous_familiar']:
                        keys = list()
                        data = dict()
                        if cats_one == cats_two:
                            if k in ['place', 'familiar']:
                                continue
                            for k_two in cats_two.split('_'):
                                rel_keys = [k for k in erp_data.keys() if k_two in k]
                                keys.append(rel_keys)
                            new_k = cats_one
                        else:
                            for k_two in cats_two.split('_'):
                                if 'person' in cats_one:
                                    key = '_'.join([k, k_two])
                                else:
                                    key = '_'.join([k_two, k])
                                keys.append(key)
                                data[key] = erp_data[key]
                            assert len(keys) == 2
                            new_k = k

                        results_time_points = list()
                        for time_window_idxs in time_windows:
                            current_erp_data = {k : numpy.average(numpy.array(v)[:, :, time_window_idxs], axis=2) for k, v in erp_data.items()}
                            #for key in keys:
                            if type(keys[0]) == list:
                                setup_data = [numpy.average([current_erp_data[single_k] for single_k in key], axis=0) for key in keys]
                                #import pdb; pdb.set_trace()
                                #label = ' - '.join([keys[0][0].split('_')[0], keys[0][0].split('_')[1]])
                                label = cats_two.replace('_', ' - ')
                            else:
                                setup_data = [current_erp_data[k] for k in keys]
                                label = ' - '.join(keys)
                            assert len(setup_data) == 2
                            setup_data = numpy.subtract(setup_data[0], setup_data[1])
                            results_time_points.append(setup_data)
                        results_time_points = numpy.array(results_time_points)
                        results_time_points = numpy.swapaxes(results_time_points, 0, 1)
                        t_stats, _, \
                        p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
                                results_time_points,
                                #tail=1, 
                                adjacency=mne_adj_matrix, 
                                threshold=dict(start=0, step=0.2), 
                                n_jobs=os.cpu_count()-1, 
                                )
                        print('Minimum p-value for {}: {}'.format(args.input_target_model, min(p_values)))

                        significance = .05
                        original_shape = t_stats.shape
                        avged_subjects = numpy.average(results_time_points, axis=0)
                        assert avged_subjects.shape == original_shape
                        significance = 0.05

                        reshaped_p = p_values.copy()
                        reshaped_p[reshaped_p>=significance] = 1.0
                        reshaped_p = reshaped_p.reshape(original_shape).T

                        reshaped_p = p_values.copy()
                        reshaped_p = reshaped_p.reshape(original_shape).T


                        #relevant_times
                        #tmin = eeg.times[0]
                        tmin = 0.
                        sfreq = 10
                        info = mne.create_info(
                                ch_names=[v for k, v in clusters.index_to_code.items()],
                                sfreq=sfreq,
                                ch_types='eeg',
                                )

                        #evoked = mne.EvokedArray(reshaped_p, info=info, tmin=tmin)
                        evoked = mne.EvokedArray(
                                                 avged_subjects.T, 
                                                 info=info, 
                                                 tmin=tmin,
                                                 )

                        montage = mne.channels.make_standard_montage('biosemi128')
                        evoked.set_montage(montage)
                        output_file = os.path.join(
                                            output_folder, 
                                            'erp_scalp_{}.jpg'.format(
                                                                      new_k,
                                                                      )
                                            )

                        ### Writing to txt
                        channels = evoked.ch_names
                        assert isinstance(channels, list)
                        assert len(channels) == reshaped_p.shape[0]
                        #assert len(times) == reshaped_p.shape[-1]
                        assert reshaped_p.shape[-1] == 8
                        txt_path = output_file.replace('.jpg', '.txt')

                        with open(txt_path, 'w') as o:
                            o.write('Time\tElectrode\tp-value\tt-value\n')
                            for t_i in range(reshaped_p.shape[-1]):
                                time = t_i
                                for c_i in range(reshaped_p.shape[0]):
                                    channel = channels[c_i]
                                    p = reshaped_p[c_i, t_i]
                                    p_value = reshaped_p[c_i, t_i]
                                    t_value = t_stats.T[c_i, t_i]
                                    o.write('{}\t{}\t{}\t{}\n'.format(time, channel, p_value, t_value))

                        title = 'Searchlight for ERP: {}'.format(
                                label
                                               )

                        evoked.plot_topomap(ch_type='eeg', 
                                time_unit='s', 
                                times=evoked.times,
                                ncols='auto',
                                nrows='auto', 
                                #vmax=vmax,
                                #vmin=0.,
                                scalings={'eeg':1.}, 
                                #cmap='Spectral_r',
                                mask=reshaped_p<=significance,
                                mask_params=dict(
                                                 marker='o', 
                                                 markerfacecolor='black', 
                                                 markeredgecolor='black',
                                                 linewidth=0, 
                                                 markersize=4,
                                                 ),
                                #colorbar=False,
                                size = 3.,
                                title=title,
                                )

                        pyplot.savefig(output_file, dpi=600)
                        #pyplot.savefig(output_file.replace('jpg', 'svg'), dpi=600)
                        pyplot.clf()
