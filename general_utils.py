import argparse
import geopy
import numpy
import os
import re
import scipy
import sklearn

from geopy import distance
from scipy import spatial, stats
from skbold.preproc import ConfoundRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR

def return_baseline(args):

    if args.mapping_direction == 'correlation' or args.evaluation_method == 'correlation':
        random_baseline=0.
    else:
        '''
        if args.experiment_id == 'one':
            if args.input_target_model == 'fine_category':
                if args.semantic_category_one == 'all':
                    random_baseline = 0.125
                else:
                    random_baseline = 0.25
            else:
                random_baseline = 0.5
        elif args.experiment_id == 'two':
            random_baseline = 0.5
        '''
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
                         ### experiment one
                         'famous', 
                         'familiar',
                         ### experiment two
                         'individual',
                         'category',
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
                        choices=['one', 'two', 'pilot'], 
                        required=True,
                        help='Which experiment?'
                        )
    parser.add_argument(
                        '--language', 
                        choices=['it', 'en'], 
                        required=True
                        )
    parser.add_argument(
                        '--average', 
                        type=int, choices=list(range(25))+[-12, -36], 
                        default=24, 
                        help='How many ERPs to average?'
                        )
    parser.add_argument('--analysis', 
                        type=str,
                        choices=[
                                 'time_resolved',
                                 'searchlight',
                                 'whole_trial',
                                 'temporal_generalization',
                                 ]
                        )
    parser.add_argument('--mapping_model', 
                        type=str,
                        choices=[
                                 'ridge',
                                 'support_vector',
                                 'rsa',
                                 ]
                        )
    parser.add_argument('--mapping_direction', 
                        type=str,
                        choices=[
                                 'encoding',
                                 'decoding',
                                 'correlation',
                                 ]
                        )

    parser.add_argument('--temporal_resolution', choices=[4, 5, 10, 25, 33, 50, 75, 100],
                        type=int, required=True)

    parser.add_argument('--data_kind', choices=[
                            'erp', 
                            'alpha',
                            'alpha_phase', 
                            'beta', 
                            'lower_gamma', 
                            'higher_gamma', 
                            'delta', 
                            'theta',
                            'theta_phase',
                            'D26',
                            'B16',
                            'C22',
                            'C23',
                            'C11',
                            ### ATLs
                            'bilateral_anterior_temporal_lobe', 
                            'left_atl', 
                            'right_atl', 
                            ### Lobes
                            'left_frontal_lobe', 'right_frontal_lobe', 'bilateral_frontal_lobe',
                            'left_temporal_lobe', 'right_temporal_lobe', 'bilateral_temporal_lobe',
                            'left_parietal_lobe', 'right_parietal_lobe', 'bilateral_parietal_lobe',
                            'left_occipital_lobe', 'right_occipital_lobe', 'bilateral_occipital_lobe',
                            'left_limbic_system', 'right_limbic_system', 'bilateral_limbic_system',
                            ### Networks
                            'language_network', 
                            'general_semantics_network',
                            'default_mode_network', 
                            'social_network', 
                            ], 
                        required=True, 
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
                                 'ceiling',
                                 ### Category
                                 'coarse_category',
                                 'famous_familiar',
                                 'fine_category',
                                 'individuals',
                                 'sex',
                                 'place_type',
                                 'occupation',
                                 'location',
                                 # orthography
                                 'orthography',
                                 'word_length',
                                 'syllables',
                                 # frequency
                                 'frequency',
                                 'log_frequency',
                                 # norms
                                 'imageability',
                                 'familiarity',
                                 'concreteness_sentence',
                                 'perceptual_sentence',
                                 'affective_sentence',
                                 'imageability_sentence',
                                 # amount of knowledge
                                 'sentence_lengths',
                                 # static language models
                                 # Static
                                 'w2v',
                                 'w2v_sentence',
                                 # Static + knowledge-aware
                                 'wikipedia2vec',
                                 'wikipedia2vec_sentence',
                                 # Knowledge-only
                                 'transe', 
                                 # contextualized
                                 'BERT_large',
                                 'MBERT', 
                                 'xlm-roberta-large',
                                 'ITGPT2',
                                 'gpt2-large',
                                 'LUKE_large',
                                 ],
                        help='Which computational model to use for decoding?'
                        )
    parser.add_argument(
                        '--evaluation_method', 
                        default='pairwise',
                        choices=['pairwise', 'correlation'],
                        help='Which evaluation method to use for decoding?'
                        )
    parser.add_argument(
                        '--searchlight_spatial_radius', 
                        choices=[
                                 ### 30mm radius, used in
                                 ### Collins et al. 2018, NeuroImage, Distinct neural processes for the perception of familiar versus unfamiliar faces along the visual hierarchy revealed by EEG
                                 ### Su et al., Optimising Searchlight Representational Similarity Analysis (RSA) for EMEG
                                 'large_distance',
                                 ### 20mm radius, used in 
                                 ### Su et al. 2014, Mapping tonotopic organization in human temporal cortex: representational similarity analysis in EMEG source space. Frontiers in neuroscience
                                 'small_distance',
                                 ### 5 closest electrodes
                                 ### Graumann et al. 2022, The spatiotemporal neural dynamics of object location representations in the human brain, nature human behaviour
                                 'fixed'
                                 ], 
                        required=True
                        )
    parser.add_argument(
                        '--searchlight_temporal_radius', 
                        choices=[
                                 ### 100ms radius
                                 'large',
                                 'medium',
                                 ### 50ms radius
                                 'small',
                                 ], 
                        required=True
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
                      action='store_true',
                      )
    args = parser.parse_args()

    check_args(args)
    
    return args

def check_args(args):
    ### checking inconsistencies in the args
    marker = False
    ### experiment one
    if args.experiment_id == 'one':
        if args.semantic_category_two in ['famous', 'familiar']:
            marker = True
            message = 'experiment two does not distinguish between famous and familiar!'
    ### experiment two 
    if args.experiment_id == 'two':
        if args.semantic_category_two in ['individual', 'category']:
            marker = True
            message = 'experiment two does not distinguish between individuals and categories!'
        if args.semantic_category_two in ['familiar', 'all']:
            if args.input_target_model in ['log_frequency', 'frequency']:
                marker = True
                message = 'frequency is not available for familiar entities!'
    if args.input_target_model == 'coarse_category' and args.semantic_category_one in ['person', 'place']:
        marker = True
        message = 'wrong model and semantic category!'
    if args.input_target_model == 'famous_familiar' and args.semantic_category_two in ['famous', 'familiar']:
        marker = True
        message = 'wrong model and semantic category!'
    if args.mapping_model in ['ridge', 'support_vector'] and args.mapping_direction == 'correlation':
        marker = True
        message = 'no correlation for ridge/support vector!'
    if args.mapping_model in ['ridge', 'support_vector'] and args.evaluation_method == 'correlation' and args.mapping_direction == 'decoding':
        if args.input_target_model in ['coarse_category', 'famous_familiar', 'word_length', 'orthography', 'sentence_lengths', 'log_frequency', 'imageability', 'familiarity', 'frequency', 'fine_category']:
            marker = True
            message = 'impossible to evaluate decoding with correlation for {}'.format(args.input_target_model)
    if args.input_target_model == 'sex':
        if args.semantic_category_one in ['place', 'all']:
            marker = True
            message = 'No sex for places!'
        if args.semantic_category_two in ['category']:
            marker = True
            message = 'No sex for categories!'
    if args.input_target_model == 'location':
        if args.semantic_category_one in ['person', 'all']:
            marker = True
            message = 'No location for places!'
        if args.semantic_category_two in ['category', 'all']:
            marker = True
            message = 'No location for categories!'
        
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

        if args.mapping_direction == 'encoding':
            if type(train_model[0]) in [int, float, numpy.float64]:
                train_model = train_model.reshape(-1, 1)
                test_model = test_model.reshape(-1, 1)
            classifier.fit(train_model, train_brain)
            test_target = test_brain.copy()
            test_input = classifier.predict(test_model)

        elif args.mapping_direction == 'decoding':
            classifier.fit(train_brain, train_model)
            test_target = test_model.copy()
            test_input = classifier.predict(test_brain)

    if args.evaluation_method == 'pairwise':

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

    elif args.evaluation_method == 'correlation':
        correct = 0.
        for idx_one, idx_two in [(0, 0), (1, 1)]:
            #print(correct)
            if type(test_target[0]) in [int, float, numpy.float64]:
                correct += 1 - abs(test_input[idx_one]-test_target[idx_two])
            else:
                correct += scipy.stats.pearsonr(test_input[idx_one], test_target[idx_two])[0]
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
            elif str(type(test_model[0])) in ["<class 'numpy.ndarray'>"] and args.input_target_model == 'location':
                test_model = [[24901 - geopy.distance.distance(tst, tr).miles for tr in train_model] for tst in test_model]
            ### for vectors, we use correlation
            ### NB: the higher the value, the more similar the items!
            else:
                test_model = [[scipy.stats.pearsonr(tst, tr)[0] for tr in train_model] for tst in test_model]
            #test_model = [model_sims[tuple(sorted([tr, tst]))] for tr in train_trigs for tst in test_trigs]

        ### rsa decoding/encoding
        if args.mapping_direction in ['encoding', 'decoding']:
            score = evaluate_pairwise(args, train_brain, test_brain, train_model, test_model, train_lengths, test_lengths)
            scores.append(score)

        ### RSA
        elif args.mapping_direction == 'correlation':
            #eeg_sims = [1. - scipy.stats.pearsonr(test_current_eeg[batch_stim], train_current_eeg[k_two])[0] for k_two in train_stimuli]
            for curr_test_brain, curr_test_model in zip(test_brain, test_model):

                eeg_sims = [scipy.stats.pearsonr(curr_test_brain, curr_train_brain)[0] for curr_train_brain in train_brain]
                model_sims = [scipy.stats.pearsonr(curr_test_model, curr_train_model)[0] for curr_train_model in train_model]
                corr = scipy.stats.pearsonr(eeg_sims, model_sims)[0]
                scores.append(corr)

    corr = numpy.average(scores)

    return corr

def prepare_folder(args):

    out_path = os.path.join(
                            'results', 
                            args.experiment_id, 
                            args.data_kind, 
                            args.analysis,
                            '{}_{}'.format(args.mapping_model, args.evaluation_method),
                            args.mapping_direction,
                            '{}ms'.format(args.temporal_resolution),
                            'average_{}'.format(args.average),
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
                             'syllables',
                             # norms
                             'imageability',
                             'familiarity',
                             # categories
                             'coarse_category',
                             'famous_familiar',
                             'fine_category',
                             'sex',
                             'location',
                             'individuals',
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

class ColorblindPalette:

    def __init__(self):
        self.black = (0, 0, 0)
        self.orange = (.9, .6, 0)
        self.celeste = (.35, .7, .9)
        self.green = (0, .6, .5)
        self.yellow = (.95, .9, .25)
        self.blue = (0, .45, .7)
        self.red = (.8, .4, 0)
        self.purple = (.8, .6, .7)

def colors_mapper():
   colors_dict = {
           ('all', 'all') : 2,
           ('all', 'famous') : 0,
           ('all', 'familiar') : 1,
           ('person', 'all') : 2,
           ('place', 'all') : 3,
           }
   return colors_dict

def read_colors(alt=False):

    file_path = 'data/12.color.blindness.palette.txt'
    assert os.path.exists(file_path)
    with open(file_path) as i:
        lines = [re.sub('\s+', r'\t', l.strip()).split('\t') for l in i.readlines()][10:]
    if alt:
        lines = lines[1::2]
    else:
        lines = lines[::2]
    assert len(lines) == 12

    colors_dict = {l_i : (int(l[3])/255, int(l[4])/255, int(l[5])/255) for l_i, l in enumerate(lines)}
    ### grouped from 'darker' to 'lighter'
    groups = {
               0 : [4, 0, 2, 7],
               1 : [5, 1, 6, 3],
               2 : [8, 9, 10, 11],
               }

    grouped_colors = {g : [colors_dict[idx] for idx in g_v] for g, g_v in groups.items()}

    return grouped_colors

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
