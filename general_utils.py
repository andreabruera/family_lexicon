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
                                 'sentence_lengths',
                                 'orthography',
                                 'imageability',
                                 'familiarity',
                                 'word_length',
                                 'frequency',
                                 'log_frequency',
                                 'syllables',
                                 'valence_individuals',
                                 'arousal_individuals',
                                 'concreteness_individuals',
                                 'imageability_individuals',
                                 'perceptual_individuals',
                                 'affective_individuals',
                                 ### English
                                 # Contextualized
                                 'BERT_base_en_sentence', 'BERT_large_en_sentence',
                                 'BERT_base_en_mentions', 'BERT_large_en_mentions',
                                 'ELMO_small_en_mentions', 'ELMO_original_en_mentions',
                                 ### bert large
                                 'BERT_large',
                                 'BERT_large_individuals',
                                 'BERT_large_random',
                                 'BERT_large_gold',
                                 'BERT_large_gold_0-150ms',
                                 'BERT_large_gold_150-300ms',
                                 'BERT_large_gold_300-500ms',
                                 'BERT_large_gold_500-800ms',
                                 'BERT_large_individuals_300-500ms',
                                 'BERT_large_model_300-500ms',
                                 'BERT_large_model_500-800ms',
                                 'BERT_large_individuals_500-800ms',
                                 'BERT_large_random_300-500ms',
                                 ### xlm roberta large
                                 'xlm-roberta-large',
                                 'xlm-roberta-large_individuals',
                                 'xlm-roberta-large_all',
                                 'xlm-roberta-large_one',
                                 'xlm-roberta-large_random',
                                 'xlm-roberta-large_gold_0-150ms',
                                 'xlm-roberta-large_gold_300-500ms',
                                 'xlm-roberta-large_model_300-500ms',
                                 'xlm-roberta-large_gold_500-800ms',
                                 ### itgpt2
                                 'ITGPT2',
                                 'ITGPT2_individuals',
                                 'ITGPT2_random',
                                 'ITGPT2_gold_0-150ms',
                                 'ITGPT2_gold_300-500ms',
                                 'ITGPT2_individuals_300-500ms',
                                 'ITGPT2_individuals_500-800ms',
                                 'ITGPT2_model_300-500ms',
                                 'ITGPT2_model_500-800ms',
                                 'ITGPT2_gold_500-800ms',
                                 ### gpt2 large
                                 'gpt2-large',
                                 'gpt2-large_individuals',
                                 'gpt2-large_random',
                                 'gpt2-large_gold_0-150ms',
                                 'gpt2-large_gold_150-300ms',
                                 'gpt2-large_gold_300-500ms',
                                 'gpt2-large_model_300-500ms',
                                 'gpt2-large_gold_500-800ms',
                                 # Contextualized + knowledge-aware
                                 'BERT_base_en_sentence', 'BERT_large_en_sentence',
                                 'LUKE_base_en_mentions', 'LUKE_large_en_mentions',
                                 # Static
                                 'w2v', \
                                 'w2v_sentence_individuals',
                                 'wikipedia2vec_sentence_individuals',
                                 # Static + knowledge-aware
                                 'wikipedia2vec', \
                                 # Knowledge-only
                                 'transe', 
                                 ### Italian
                                 'LUKE_large',
                                 'SPANBERT_large', 
                                 'MBERT', 
                                 'ITBERT',
                                 'it_w2v',
                                 'it_wikipedia2vec',
                                 'ITGPT2medium',
                                 'BERT_base_it_mentions',
                                 ### Ceiling
                                 'ceiling',
                                 ### Category
                                 'coarse_category',
                                 'famous_familiar',
                                 'fine_category',
                                 'mixed_category',
                                 'individuals',
                                 'gender',
                                 'place_type',
                                 'occupation',
                                 'location',
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
    #if 'gold' in args.input_target_model and args.average!=-36:
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


    test_samples = list()
    test_true = list()

    test_lengths = list()

    for trig in split:
        #if args.input_target_model not in ['coarse_category', 'fine_category', 'famous_familiar'] or args.mapping_model == 'rsa':
        if 1 == 1:
        #if args.mapping_direction != 'correlation':
            trig = experiment.trigger_to_info[trig][0]
            vecs = comp_vectors[trig]
        erps = eeg[trig]
        if not isinstance(erps, list):
            erps = [erps]
            #if args.input_target_model not in ['coarse_category', 'fine_category', 'famous_familiar'] or args.mapping_model == 'rsa':
            if 1 == 1:
            #if args.mapping_direction != 'correlation':
                vecs = [vecs]
        #if args.input_target_model not in ['coarse_category', 'fine_category', 'famous_familiar'] or args.mapping_model == 'rsa':
        if 1 == 1:
        #if args.mapping_direction != 'correlation':
            test_samples.extend(vecs)
            test_true.extend(erps)
            test_lengths.append(len(trig))
        else:
            test_samples.extend(erps)
            test_true.extend([experiment.trigger_to_info[trig][cat_index] for i in range(len(erps))])
            test_lengths.append(len(experiment.trigger_to_info[trig][0]))

    train_samples = list()
    train_true = list()

    train_lengths = list()

    for k, erps in eeg.items():
        #if args.input_target_model not in ['coarse_category', 'fine_category', 'famous_familiar'] or args.mapping_model == 'rsa':
        if 1 == 1:
        #if args.mapping_direction != 'correlation':
            k = {v[0] : k for k, v in experiment.trigger_to_info.items()}[k]
        if not isinstance(erps, list):
            erps = [erps]
        if k not in split:
            if 1 == 1:
            #if args.input_target_model not in ['coarse_category', 'fine_category', 'famous_familiar'] or args.mapping_model == 'rsa':
            #if args.mapping_direction != 'correlation':
                k = experiment.trigger_to_info[k][0]
                vecs = [comp_vectors[k]]
                train_samples.extend(vecs)
                train_true.extend(erps)
                train_lengths.append(len(k))
            else:
                train_samples.extend(erps)
                train_true.extend([experiment.trigger_to_info[k][cat_index] for i in range(len(erps))])
                train_lengths.append(len(experiment.trigger_to_info[k][0]))

    test_samples = numpy.array(test_samples, dtype=numpy.float64)
    train_samples = numpy.array(train_samples, dtype=numpy.float64)

    if 0 == 1:
        train_true = LabelEncoder().fit_transform(train_true)
        test_true = LabelEncoder().fit_transform(test_true)
        
        #train_labels = len(list(set(train_true)))
    else:
        train_true = numpy.array(train_true, dtype=numpy.float64)
        test_true = numpy.array(test_true, dtype=numpy.float64)

        assert train_samples.shape[0] == len(eeg.keys())-2
        assert len(test_samples) == 2

    train_lengths = numpy.array(train_lengths)
    test_lengths = numpy.array(test_lengths)

    assert train_lengths.shape[0] == train_samples.shape[0]
    assert test_lengths.shape[0] == test_samples.shape[0]

    return train_true, test_true, train_samples, test_samples, train_lengths, test_lengths

def correct_for_length(args, train_true, train_lengths, test_true, test_lengths):
    '''
    ### only train
    cfr = ConfoundRegressor(confound=train_lengths, X=train_true.copy())
    cfr.fit(train_true)
    train_true = cfr.transform(train_true)
    '''
    ### train + test
    correction_lengths = numpy.hstack([train_lengths, test_lengths])
    correction_data = numpy.vstack([train_true, test_true])
    cfr = ConfoundRegressor(confound=correction_lengths, X=correction_data)
    cfr.fit(train_true)
    train_true = cfr.transform(train_true)
    test_true = cfr.transform(test_true)

    return train_true, test_true

def evaluate_pairwise(args, train_true, test_true, train_samples, test_samples, train_lengths, test_lengths):

    if args.mapping_model == 'rsa':

        if args.mapping_direction == 'encoding':
            ### Encoding targets (brain images)
            test_samples = [numpy.sum([t*corrs[t_i] for t_i, t in enumerate(train_true)], axis=0) for corrs in test_samples]

        elif args.mapping_direction == 'decoding':
            ### test_samples = model correlations
            test_true = [[scipy.stats.pearsonr(tst, tr)[0] for tr in train_true] for tst in test_true]

    elif args.mapping_model in ['ridge', 'support_vector']:


        ### Differentiating between binary and multiclass classifier
        if args.mapping_model == 'support_vector':
            classifier = SVR()
        elif args.mapping_model == 'ridge':
            classifier = Ridge()
            #print('loading ridge')
            #classifier = RidgeCV(alphas=(0.00005, 1, 10))
        train_samples = scipy.stats.zscore(train_samples)
        test_samples = scipy.stats.zscore(test_samples)
        train_true = scipy.stats.zscore(train_true, axis=1)
        test_true = scipy.stats.zscore(test_true, axis=1)

        if args.mapping_direction == 'encoding':
            if type(train_samples[0]) in [int, float, numpy.float64]:
                train_samples = train_samples.reshape(-1, 1)
                test_samples = test_samples.reshape(-1, 1)
            classifier.fit(train_samples, train_true)
            ### test_samples = predictions
            test_samples = classifier.predict(test_samples)

        elif args.mapping_direction == 'decoding':
            classifier.fit(train_true, train_samples)
            ### test_true = predictions
            predictions = classifier.predict(test_true)
            test_true = test_samples.copy()
            test_samples = predictions.copy()

    if args.evaluation_method == 'pairwise':

        wrong = 0.
        for idx_one, idx_two in [(0, 1), (1, 0)]:
            if type(test_samples[0]) in [int, float, numpy.float64]:
                wrong += 1 - abs(test_samples[idx_one]-test_true[idx_two])
            else:
                wrong += scipy.stats.pearsonr(test_samples[idx_one], test_true[idx_two])[0]

        correct = 0.
        for idx_one, idx_two in [(0, 0), (1, 1)]:
            if type(test_samples[0]) in [int, float, numpy.float64]:
                correct += 1 - abs(test_samples[idx_one]-test_true[idx_two])
            else:
                correct += scipy.stats.pearsonr(test_samples[idx_one], test_true[idx_two])[0]

        if correct > wrong:
            #accuracies.append(1)
            accuracy = 1
        else:
            #accuracies.append(0)
            accuracy = 0

    elif args.evaluation_method == 'correlation':
        correct = 0.
        for idx_one, idx_two in [(0, 0), (1, 1)]:
            if type(test_samples[0]) in [int, float, numpy.float64]:
                correct += 1 - abs(test_samples[idx_one]-test_true[idx_two])
            else:
                correct += scipy.stats.pearsonr(test_samples[idx_one], test_true[idx_two])[0]
        accuracy = correct/2

    return accuracy

def evaluation_round(args, experiment, current_eeg, comp_vectors):

    scores = list()

    for split in experiment.test_splits:
        train_true, test_true, train_samples, test_samples, train_lengths, test_lengths = split_train_test(args, split, current_eeg, experiment, comp_vectors)
        if type(test_samples[0]) in [int, float, numpy.float64, tuple]:
            if test_samples[0] == test_samples[1]:
                continue
        ### regress out word_length
        if args.corrected:
            train_true, test_true = correct_for_length(args, train_true, train_lengths, test_true, test_lengths)

        ### transforming model
        if args.mapping_model == 'rsa':
            if type(test_samples[0]) in [int, float, numpy.float64]:
                test_samples = [[1 - abs(tst-tr) for tr in train_samples] for tst in test_samples]
            elif str(type(test_samples[0])) in ["<class 'numpy.ndarray'>"] and args.input_target_model == 'location':
                test_samples = [[24901 - geopy.distance.distance(tst, tr).miles for tr in train_samples] for tst in test_samples]
            ### for vectors, we use correlation
            ### NB: the higher the value, the more similar the items!
            else:
                test_samples = [[scipy.stats.pearsonr(tst, tr)[0] for tr in train_samples] for tst in test_samples]

        ### rsa decoding/encoding
        if args.mapping_direction in ['encoding', 'decoding']:
            score = evaluate_pairwise(args, train_true, test_true, train_samples, test_samples, train_lengths, test_lengths)
            scores.append(score)

        ### RSA
        elif args.mapping_direction == 'correlation':
            #eeg_sims = [1. - scipy.stats.pearsonr(test_current_eeg[batch_stim], train_current_eeg[k_two])[0] for k_two in train_stimuli]
            for test_brain, test_model in zip(test_true, test_samples):

                eeg_sims = [scipy.stats.pearsonr(test_brain, train_brain)[0] for train_brain in train_true]
                model_sims = [scipy.stats.pearsonr(test_model, train_model)[0] for train_model in train_samples]
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
                             'orthography',
                             'imageability',
                             'familiarity',
                             'word_length',
                             'syllables',
                             'individuals',
                             'place_type',
                             'occupation',
                             'gender',
                             'location',
                             'nationality',
                             'distance_from_trieste',
                             ### Ceiling
                             'ceiling',
                             ### Category
                             'coarse_category',
                             'famous_familiar',
                             'fine_category',
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
