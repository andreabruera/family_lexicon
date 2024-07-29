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
from tqdm import tqdm

from io_utils import ExperimentInfo, LoadEEG

def read_color(args):
    if args.comparison:
        color_one = 'lightskyblue'
        color_two = 'slategrey'
        time_resolved_color = 'black'
    else:
        if args.input_target_model in ['word_length', 'orthography']:
            color_two = 'black'
            if args.input_target_model == 'word_length':
                color_one = 'black'
            else:
                color_one = 'gray'
            time_resolved_color = color_one
        else:
            ### colorblind palettes retrieved from
            #https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40
            ### Wong colorblind palette
            if args.semantic_category_one == 'all':
                if 'xlm' in args.input_target_model:
                    color_one = 'mediumseagreen'
                    color_two = 'darkgreen'
                elif 'w2v' in args.input_target_model:
                    color_one = 'goldenrod'
                    color_two = 'saddlebrown'
                time_resolved_color = color_one
            ### for XLM: IBM palette
            ### fow w2v: Tol palette
            elif args.semantic_category_one == 'person':
                if args.semantic_category_two == 'famous':
                    if 'xlm' in args.input_target_model:
                        color_one = 'pink'
                        color_two = 'mediumvioletred'
                        time_resolved_color = color_two
                    elif 'w2v' in args.input_target_model:
                        color_one = 'palevioletred'
                        color_two = 'mediumvioletred'
                        time_resolved_color = color_one
                elif args.semantic_category_two == 'familiar':
                    if 'xlm' in args.input_target_model:
                        #color_two = 'indigo'
                        color_one = 'lavender'
                        color_two = 'slateblue'
                        time_resolved_color = color_two
                    elif 'w2v' in args.input_target_model:
                        #color_two = 'darkblue'
                        color_one = 'lightcyan'
                        color_two = 'lightseagreen'
                        time_resolved_color = color_two
            elif args.semantic_category_one == 'place':
                if args.semantic_category_two == 'famous':
                    if 'xlm' in args.input_target_model:
                        color_one = 'orange'
                        color_two = 'chocolate'
                        time_resolved_color = color_one
                    elif 'w2v' in args.input_target_model:
                        color_one = 'khaki'
                        color_two = 'olive'
                        time_resolved_color = 'darkkhaki'
                elif args.semantic_category_two == 'familiar':
                    if 'xlm' in args.input_target_model:
                        color_one = 'cornflowerblue'
                        color_two = 'mediumblue'
                        time_resolved_color = color_one
                    elif 'w2v' in args.input_target_model:
                        color_one = 'lightskyblue'
                        color_two = 'steelblue'
                        time_resolved_color = color_one
    return color_one, color_two, time_resolved_color

def return_baseline(args):

    random_baseline = 0.

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
                        '--experiment_id',
                        choices=['two'],
                        default='two',
                        help='Which experiment?'
                        )
    parser.add_argument(
                        '--language',
                        choices=['it'],
                        default='it',
                        )
    parser.add_argument(
                        '--average',
                        type=int, choices=[24],
                        default=24,
                        help='How many ERPs to average?'
                        )
    parser.add_argument('--analysis',
                        type=str,
                        choices=[
                                 'time_resolved',
                                 'searchlight',
                                 ]
                        )
    parser.add_argument('--mapping_model',
                        type=str,
                        choices=[
                                 'rsa',
                                 ],
                        default='rsa',
                        )
    parser.add_argument('--mapping_direction',
                        type=str,
                        choices=[
                                 'encoding',
                                 ],
                        default='encoding',
                        )

    parser.add_argument(
                        '--temporal_resolution',
                        choices=[4, 5, 10, 25, 33, 50, 75, 100],
                        type=int,
                        default=5,
                        )

    parser.add_argument('--data_kind', choices=[
                            'erp',
                            ],
                        default='erp',
                        help='Time-frequency, ERP or source analyses?'
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
                        default=True,
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
                                 'ceiling',
                                 'w2v_sentence',
                                 'xlm-roberta-large',
                                 'orthography',
                                 'word_length',
                                 ],
                        help='Which computational model to use for decoding?'
                        )
    parser.add_argument(
                        '--evaluation_method',
                        default='correlation',
                        choices=['correlation'],
                        help='Which evaluation method to use for decoding?'
                        )
    parser.add_argument(
                        '--searchlight_spatial_radius',
                        choices=[
                                 ### 30mm radius, used in
                                 ### Collins et al. 2018, NeuroImage, Distinct neural processes for the perception of familiar versus unfamiliar faces along the visual hierarchy revealed by EEG
                                 ### Su et al., Optimising Searchlight Representational Similarity Analysis (RSA) for EMEG
                                 'large_distance',
                                 ],
                        default='large_distance',
                        )
    parser.add_argument(
                        '--searchlight_temporal_radius',
                        choices=[
                                 ### 100ms radius
                                 'large',
                                 ],
                        default='large',
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
    parser.add_argument(
                      '--across_subjects',
                      default=False,
                      action='store_true',
                      )
    args = parser.parse_args()

    check_args(args)

    return args

def check_args(args):
    ### checking inconsistencies in the args
    marker = False
    ### experiment two
    if args.experiment_id == 'two':
        if args.semantic_category_two in ['familiar', 'all']:
            if args.input_target_model in ['log_frequency', 'frequency']:
                marker = True
                message = 'frequency is not available for familiar entities!'
    if marker:
        raise RuntimeError(message)

def split_train_test(args, split, eeg, experiment, comp_vectors):

    ### Selecting the relevant index
    ### for the trigger_to_info dictionary
    if args.input_target_model == 'coarse_category':
        cat_index = 1
    if args.input_target_model == 'fine_category':
        cat_index = 2
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

        if args.mapping_direction == 'encoding':
            ### Encoding targets (brain images)
            test_input = [numpy.sum([t*corrs[t_i] for t_i, t in enumerate(train_brain)], axis=0) for corrs in test_model]
            #test_input = [numpy.sum([t*corr for t in train_brain], axis=0) for corr in test_model]
            test_target = test_brain.copy()

        elif args.mapping_direction == 'decoding':
            test_input = [[scipy.stats.pearsonr(tst, tr)[0] for tr in train_brain] for tst in test_brain]
            test_target = test_model.copy()

    if args.evaluation_method == 'correlation':
        correct = 0.
        for idx_one, idx_two in [(0, 0), (1, 1)]:
            if type(test_target[0]) in [int, float, numpy.float64]:
                correct += 1 - abs(test_input[idx_one]-test_target[idx_two])
            else:
                correct += scipy.stats.spearmanr(test_input[idx_one], test_target[idx_two])[0]
        accuracy = correct/2

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
            if args.mapping_direction == 'encoding' and args.evaluation_method == 'correlation':
                train_brain, test_brain = remove_average(train_brain, test_brain)

        ### transforming model to pairwise if required
        if args.mapping_model == 'rsa':
            if type(test_model[0]) in [int, float, numpy.float64]:
                test_model = [[1 - abs(tst-tr) for tr in train_model] for tst in test_model]
            ### for vectors, we use correlation
            ### NB: the higher the value, the more similar the items!
            else:
                test_model = [[scipy.stats.pearsonr(tst, tr)[0] for tr in train_model] for tst in test_model]

        ### rsa decoding/encoding
        if args.mapping_direction in ['encoding', 'decoding']:
            score = evaluate_pairwise(args, train_brain, test_brain, train_model, test_model, train_lengths, test_lengths)
            scores.append(score)

    corr = numpy.average(scores)

    return corr

def prepare_folder(args):

    out_path = os.path.join(
                            'results',
                            args.analysis,
                            args.semantic_category_one,
                            args.semantic_category_two,
                             )
    os.makedirs(out_path, exist_ok=True)

    return out_path

def prepare_file(args, n):

    lang_agnostic_models = [
                             ### Ceiling
                             'ceiling',
                             # orthography
                             'orthography',
                             'word_length',
                             ]

    if args.input_target_model in lang_agnostic_models:
        language_agnostic = True
    else:
        language_agnostic = False

    correction = 'corrected' if args.corrected else 'uncorrected'
    file_path = 'sub_{:02}_{}_{}.scores'.format(n, correction, args.input_target_model)

    if not language_agnostic:
        file_path = file_path.replace('.scores', '_{}.scores'.format(args.language))

    if args.analysis == 'searchlight':
        file_path = file_path.replace(
                '.scores',
                '_spatial_{}_temporal_{}.scores'.format(
                       args.searchlight_spatial_radius,
                       args.searchlight_temporal_radius
                       )
                )
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
