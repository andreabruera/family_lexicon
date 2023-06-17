import os
import mne
import pandas
import random
import re
import collections
import nilearn
import numpy
import matplotlib
import itertools
import random
import scipy

from matplotlib import pyplot
from nilearn import image
from scipy import stats
from tqdm import tqdm

#from lab.utils import read_words, read_trigger_ids, select_words
#from searchlight import SearchlightClusters

def tfr_frequencies(args):
    ### Setting up each frequency band
    if 'alpha' in args.data_kind:
        frequencies = list(range(8, 15))
    elif args.data_kind == 'beta':
        frequencies = list(range(14, 31))
    elif args.data_kind == 'lower_gamma':
        frequencies = list(range(30, 51))
    elif args.data_kind == 'higher_gamma':
        frequencies = list(range(50, 81))
    elif args.data_kind == 'delta':
        frequencies = list(range(1, 5)) + [0.5]
    elif 'theta' in args.data_kind:
        frequencies = list(range(4, 9))
    #elif args.data_kind == ('erp'):
    else:
        frequencies = 'na'
    frequencies = numpy.array(frequencies)

    return frequencies

class ExperimentInfo:

    def __init__(self, args, subject=1):
        
        self.experiment_id = args.experiment_id
        self.analysis = args.analysis
        self.mapping_model = args.mapping_model
        self.mapping_direction = args.mapping_model
        self.semantic_category_one = args.semantic_category_one
        self.semantic_category_two = args.semantic_category_two
        self.data_folder = args.data_folder
        self.corrected = args.corrected
        self.runs = 24
        self.subjects = 33
        self.current_subject = subject
        self.eeg_paths = self.generate_eeg_paths(args)
        self.events_log, self.trigger_to_info = self.read_events_log()
        self.test_splits = self.generate_test_splits()

    def generate_eeg_paths(self, args):
        eeg_paths = dict()

        for s in range(1, self.subjects+1):

            if args.data_kind in ['erp', 'D26', 'B16','C22', 'C23', 'C11','alpha', 'alpha_phase', 'beta', 'delta', 'theta', 'theta_phase', 'lower_gamma', 'higher_gamma']:
                fold = 'derivatives'
                sub_path = os.path.join(
                                        self.data_folder,
                                        fold,
                                        'sub-{:02}'.format(s),
                                        'sub-{:02}_task-namereadingimagery_eeg-epo.fif.gz'.format(s)
                                        )
                #print(sub_path)

            elif args.data_kind in [
                                    ### ATLs
                                    'bilateral_anterior_temporal_lobe', 'left_atl', 'right_atl', 
                                    ### Lobes
                                    'left_frontal_lobe', 'right_frontal_lobe', 'bilateral_frontal_lobe',
                                    'left_temporal_lobe', 'right_temporal_lobe', 'bilateral_temporal_lobe',
                                    'left_parietal_lobe', 'right_parietal_lobe', 'bilateral_parietal_lobe',
                                    'left_occipital_lobe', 'right_occipital_lobe', 'bilateral_occipital_lobe',
                                    'left_limbic_system', 'right_limbic_system', 'bilateral_limbic_system',
                                    ### Networks
                                    'language_network', 'general_semantics_network',
                                    'default_mode_network', 'social_network', 
                                    ]:

                fold = 'reconstructed'
                sub_path = os.path.join(
                                        self.data_folder,
                                        fold,
                                        'sub-{:02}'.format(s),
                                        )

            assert os.path.exists(sub_path) == True
            eeg_paths[s] = sub_path
        return eeg_paths

    def read_events_log(self):

        full_log = dict()

        for s in range(1, self.subjects+1):
            events_path = os.path.join(self.data_folder,
                                    'derivatives',
                                    'sub-{:02}'.format(s),
                                    'sub-{:02}_task-namereadingimagery_events.tsv'.format(s))
            assert os.path.exists(events_path) == True
            with open(events_path) as i:
                sub_log = [l.strip().split('\t') for l in i.readlines()]
            sub_log = {h : [l[h_i].strip() for l in sub_log[1:]] for h_i, h in enumerate(sub_log[0])}
            n_trials = list(set([len(v) for v in sub_log.values()]))
            assert len(n_trials) == 1
            ### Initializing the dictionary
            if len(full_log.keys()) == 0:
                full_log = sub_log.copy()
                full_log['subject'] = list()
            ### Adding values
            else:
                for k, v in sub_log.items():
                    full_log[k].extend(v)
            ### adding subject
            for i in range(n_trials[0]):
                full_log['subject'].append(s)
        ### Creating the trigger-to-info dictionary
        trig_to_info = dict()
        for t_i, t in enumerate(full_log['value']):
            ### Limiting the trigger-to-info dictionary to the current subject
            if full_log['subject'][t_i] == self.current_subject:
                name = full_log['trial_type'][t_i]
                if self.experiment_id == 'two':
                    key_one = 'semantic_domain'
                    key_two = 'familiarity'
                else:
                    key_one = 'coarse_category'
                    key_two = 'fine_category'
                cat_one = full_log[key_one][t_i]
                cat_two = full_log[key_two][t_i]
                infos = [name, cat_one, cat_two]
                t = int(t)
                if t in trig_to_info.keys():
                    assert trig_to_info[t] == infos
                trig_to_info[t] = infos

        ### Filtering trig_to_info as required

        '''
        ### first distinction: people/places
        if self.semantic_category_one == 'all':
            pass
        else:
            trig_to_info = {k : v for k, v in trig_to_info.items() if v[1]==self.semantic_category_one}

        ### second subdivision famous/familiar or individuals/categories
        if self.semantic_category_two == 'all':
            pass
        else:
            if self.experiment_id == 'two':
                trig_to_info = {k : v for k, v in trig_to_info.items() if v[2]==self.semantic_category_two}
            elif self.experiment_id == 'one':
                if self.semantic_category_two == 'individual':
                    trig_to_info = {k : v for k, v in trig_to_info.items() if k<=100}
                elif self.semantic_category_two == 'category':
                    trig_to_info = {k : v for k, v in trig_to_info.items() if k>100}
        if self.experiment_id == 'one':
            if self.semantic_category_two == 'individual':
                trig_to_info = {k : v for k, v in trig_to_info.items() if k<=100}
            elif self.semantic_category_two == 'category':
                trig_to_info = {k : v for k, v in trig_to_info.items() if k>100}
        '''

        return full_log, trig_to_info

    def generate_test_splits(self):

        #all_combs = list(itertools.combinations(self.trigger_to_info.keys(), r=2))

        ### first distinction: people/places
        if self.semantic_category_one == 'all':
            relevant_trigs_one = self.trigger_to_info.keys()
        else:
            #trig_to_info = {k : v for k, v in trig_to_info.items() if v[1]==self.semantic_category_one}
            relevant_trigs_one = [k for k, v in self.trigger_to_info.items() if v[1]==self.semantic_category_one]

        ### second subdivision famous/familiar or individuals/categories
        if self.semantic_category_two == 'all':
            relevant_trigs_two = self.trigger_to_info.keys()
        else:
            if self.experiment_id == 'two':
                relevant_trigs_two = [k for k, v in self.trigger_to_info.items() if v[2]==self.semantic_category_two]
            elif self.experiment_id == 'one':
                if self.semantic_category_two == 'individual':
                    relevant_trigs_two = [k for k, v in self.trigger_to_info.items() if k<=100]
                elif self.semantic_category_two == 'category':
                    relevant_trigs_two = [k for k, v in self.trigger_to_info.items() if k>100]
        relevant_trigs = [k for k in self.trigger_to_info.keys() if k in relevant_trigs_one and k in relevant_trigs_two]
        print(relevant_trigs)

        all_combs = list(itertools.combinations(list(relevant_trigs), r=2))

        ### removing useless leave-two-out splits
        #all_combs = [c for c in all_combs if c[0] in relevant_trigs and c[1] in relevant_trigs]

        return all_combs

    def OLD_generate_test_splits(self):

        ### Experiment two, somewhat easier
        ### as there are always two classes
        if self.experiment_id == 'two':
            ### Creating for each test set two tuples, one for each class

            ### always two!
            cat_length = 8
            combinations_one_cat = list(itertools.combinations(list(range(cat_length)), 1))
            combinations = list(itertools.product(combinations_one_cat, repeat=2))
            '''
            ### Reducing stimuli to nested sub-class
            if self.semantic_category in ['people', 'places', 'famous', 'familiar']:
                cat_index = 2
                cat_length = 8
                combinations_one_cat = list(itertools.combinations(list(range(cat_length)), 1))
                combinations = list(itertools.product(combinations_one_cat, repeat=2))
            ### Using all stimuli
            else:
                cat_length = 16
                combinations_one_cat = list(itertools.combinations(list(range(cat_length)), 2))
                combinations = list(itertools.product(combinations_one_cat, repeat=2))
            '''
            ### Semantic domain
            if 'coarse' in self.analysis or self.semantic_category in ['familiar', 'famous']:
                cat_index = 1
            ### Type of familiarity
            else:
                cat_index = 2

        ### Experiment one, there we go...
        else:
            ### Transfer classification (nice 'n easy...)
            if self.entities == 'individuals_to_categories':
                ### Creating for each test set two tuples, one for each class
                if 'coarse' in self.analysis:
                    cat_index = 1
                    cat_length = 4
                    combinations_one_cat = list(itertools.combinations(list(range(cat_length)), 2))
                    combinations = list(itertools.product(combinations_one_cat, repeat=2))
                ### When looking at fine-grained categories, 
                ### there's only one test set with the 4/8 classes
                elif 'fine' in self.analysis:
                    cat_index = 2
                    cat_length = 1
                    n_classes = 8 if self.semantic_category=='all' else 4
                    combinations = [(0 for i in range(n_classes))]
            ### Standard classification
            else:
                ### People vs places
                if 'coarse' in self.analysis:
                    cat_index = 1
                    ### Using individuals only
                    if self.entities == 'individuals_only':
                        cat_length = 16
                    ### Mixing individuals and categories
                    elif self.entities == 'individuals_and_categories':
                        cat_length = 20
                    combinations_one_cat = list(itertools.combinations(list(range(cat_length)), 2))
                    combinations = list(itertools.product(combinations_one_cat, repeat=2))
                ### Fine-grained categories
                elif 'fine' in self.analysis:
                    cat_index = 2
                    if self.entities == 'individuals_only':
                        cat_length = 4
                    elif self.entities == 'individuals_and_categories':
                        cat_length = 5
                    ### Transfer classification
                    else:
                        cat_length = 1
                    ### Using all stimuli
                    if self.semantic_category == 'all':
                        combinations = list(itertools.product(list(range(cat_length)), repeat=8))
                    ### Using only a nested sub-class
                    else:
                        combinations = list(itertools.product(list(range(cat_length)), repeat=4))

        ### Getting the list of classes to be used
        cats = set([v[cat_index] for k, v in self.trigger_to_info.items()])
        ### Transfer classification requires to reduce the possible test triggers
        if self.entities == 'individuals_to_categories':
            cat_to_trigger = {cat : [t for t, info in self.trigger_to_info.items() if info[cat_index] == cat and t>100] for cat in cats}
        ### Standard classification considers them all
        else:
            cat_to_trigger = {cat : [t for t, info in self.trigger_to_info.items() if info[cat_index] == cat] for cat in cats}

        ### Just checking all's good and fine
        for k, v in cat_to_trigger.items():
            assert len(v) == cat_length

        ### Randomizing all test splits
        test_permutations = list(random.sample(combinations, k=len(combinations)))

        ### Creating test splits, and correcting them for length if required
        test_splits = list()
        for i_p, p in enumerate(test_permutations):
            triggers = list()
            ### Fine-grained categories requires a separate case
            if 'fine' in self.analysis:
                if self.semantic_category != 'all':
                    assert len(cat_to_trigger.keys()) == 4
                else:
                    assert len(cat_to_trigger.keys()) == 8
                for t_i, ts in zip(p, cat_to_trigger.values()):
                    triggers.append(ts[t_i])

                ### Checking
                if self.semantic_category != 'all':
                    assert len(triggers) == 4
                else:
                    assert len(triggers) == 8 
            ### Cases where there's only two classes
            else:
                assert len(cat_to_trigger.keys()) == 2
                for kv_i, kv in enumerate(cat_to_trigger.items()):
                    for ind in p[kv_i]:
                        triggers.append(kv[1][ind])

                ### Checking
                if self.semantic_category != 'all':
                    assert len(triggers) == 2
                else:
                    assert len(triggers) == 4 

            test_splits.append(triggers)
        ### Difference across people and places / familiar and unfamiliar
        if 'coarse' in self.analysis:

            stat_diff = [[len(n[0]) for n in self.trigger_to_info.values() if n[1]==k] for k in set([v[1] for v in self.trigger_to_info.values()])]
            stat_diff = scipy.stats.ttest_ind(stat_diff[0], stat_diff[1])
            print('Difference between lengths for people and places: {}'.format(stat_diff))

        ### Collecting average lengths for the current labels

        ### To correct both for coarse and fine-grained lengths
        ### use this line
        ### For replication of experiment one, use this line
        #if 'coarse' in args.analysis or 'fine' in args.analysis:
        current_cats = set([v[cat_index] for v in self.trigger_to_info.values()])
        cat_to_average_length = {k : numpy.average([len(n[0]) for n in self.trigger_to_info.values() if n[cat_index]==k]) for k in current_cats}
        cat_to_lengths = [[len(n[0]) for n in self.trigger_to_info.values() if n[cat_index]==k] for k in current_cats]
        stat_sig_length = scipy.stats.ttest_ind(cat_to_lengths[0], cat_to_lengths[1])
        ### Replication
        if self.experiment_id == 'one':
            cat_to_average_length = {'actor' : 14,
                                  'musician' : 9,
                                  'writer' : 13,
                                  'politician' : 13,
                                  'person' : 12, 
                                  'place' : 9,
                                  'city' : 6, 
                                  'country' : 7,
                                  "body_of_water" : 12, 
                                  'monument' : 11
                                  }
        print('Current categories and average lengths: {}'.format(cat_to_average_length))
        print('current statistical difference among the two categories: {}'.format(stat_sig_length))

        ### correction needs to take place only when
        ### difference is statistically significant
        '''

        if stat_sig_length[1] < 0.1 and self.corrected:
            if len(test_splits[0]) == 2:
                std = numpy.std([len(v[0]) for v in self.trigger_to_info.values()])
                ### simple strategy: taking test cases where distance among items
                ### is not superior to 0.5 std
                test_splits = [t for t in test_splits if abs(len(self.trigger_to_info[t[0]][0])-len(self.trigger_to_info[t[1]][0]))<=std]
                assert len(test_splits) > 1
                print('for subject {}, considering a maximum difference of {} letters'.format(self.current_subject, int(std)))

            ### using correlation
            else:
                ### Computing correlation
                split_corrs = list()
                for trigs in test_splits:
                    labels = list()
                    lengths = list()
                    ### Test set
                    for t in trigs:
                        labels.append(cat_to_average_length[self.trigger_to_info[t][cat_index]])
                        lengths.append(len(self.trigger_to_info[t][0]))
                    if len(trigs) > 2:
                        corr = list(scipy.stats.pearsonr(lengths, labels))
                    else:
                        corr = abs(lengths[0]-lengths[1])
                    split_corrs.append(corr)
                split_corrs = sorted(enumerate(split_corrs), key=lambda item : abs(item[1][0]))
                zero_corr_sets = len([t for t in split_corrs if t[1][0]==0.0])
                if len(zero_corr_sets) >= 10:
                    test_splits = random.sample([test_splits[t[0]] for t in split_corrs if t[1][0]==0.0], k=n_folds)
                else:
                    test_splits = [test_splits[t[0]] for t in split_corrs][:10]

        ### no correction
        else:
            n_folds = len(test_splits)
            test_splits = random.sample(test_splits, k=min(n_folds, len(test_splits)))
        print('using {} splits'.format(len(test_splits)))
        '''
        n_folds = len(test_splits)
        test_splits = random.sample(test_splits, k=min(n_folds, len(test_splits)))
        print('using {} splits'.format(len(test_splits)))

        return test_splits

class LoadEEG:
   
    def __init__(self, args, experiment, subject, ceiling=False):

        self.subject = subject
        self.data_path = experiment.eeg_paths[subject]
        self.experiment = experiment
        self.ceiling = ceiling
        self.trigger_to_info = experiment.trigger_to_info
        if args.data_kind in ['erp', 'D26', 'B16', 'C22', 'C23', 'C11', 'alpha', 'alpha_phase', 'beta', 'delta', 'theta', 'theta_phase', 'lower_gamma', 'higher_gamma']:
            self.full_data, self.data_dict, \
            self.times, self.all_times, \
            self.frequencies, self.data_shape = self.load_epochs(args)
        elif args.data_kind in [
                                ### ATLs
                                'bilateral_anterior_temporal_lobe', 'left_atl', 'right_atl', 
                                ### Lobes
                                'left_frontal_lobe', 'right_frontal_lobe', 'bilateral_frontal_lobe',
                                'left_temporal_lobe', 'right_temporal_lobe', 'bilateral_temporal_lobe',
                                'left_parietal_lobe', 'right_parietal_lobe', 'bilateral_parietal_lobe',
                                'left_occipital_lobe', 'right_occipital_lobe', 'bilateral_occipital_lobe',
                                'left_limbic_system', 'right_limbic_system', 'bilateral_limbic_system',
                                ### Networks
                                'language_network', 'general_semantics_network',
                                'default_mode_network', 'social_network', 
                                ]:
            self.fmri_mask = self.load_mask(args)
            self.full_data, self.data_dict, \
            self.times, self.all_times, \
            self.frequencies, self.data_shape = self.load_source(args)

    def load_mask(self, args):

        map_path = os.path.join('fmri_masks', '{}.nii'.format(args.data_kind))
        try:
            assert os.path.exists(map_path)
        except AssertionError:
            map_path = os.path.join('fmri_masks', '{}.nii.gz'.format(args.data_kind))
            assert os.path.exists(map_path)
        print('Masking {}...'.format(args.data_kind))
        map_nifti = nilearn.image.load_img(map_path)
        map_nifti = nilearn.image.binarize_img(map_nifti, threshold=0.)
        sample_f = os.path.join(self.data_path, 'trigger_1.nii.gz')
        sample_img = nilearn.image.load_img(sample_f)
        map_nifti = nilearn.image.resample_to_img(map_nifti, sample_f, interpolation='nearest')

        return map_nifti

    def load_source(self, args):

        data_dict =dict()

        for trig in self.trigger_to_info.keys():
            f = os.path.join(self.data_path, 'trigger_{}.nii.gz'.format(trig))
            img = nilearn.image.load_img(f)
            label = re.sub('^.+_|[.].+$', '', f)
            masked_data = nilearn.masking.apply_mask(img, self.fmri_mask)
            data_dict[int(label)] = masked_data.T
            data_shape = data_dict[int(label)].shape

        times = [
                -.05, .05, .15, .25, 
                 .35, .45, .55, .65,
                 .75, .85, .95, #1.05
                 ]
        full_data_dict = data_dict.copy()
        all_times = times.copy()

        frequencies = tfr_frequencies(args)

        return full_data_dict, data_dict, times, all_times, frequencies, data_shape

    def load_epochs(self, args):

        coll_data_dict = collections.defaultdict(list)
        epochs = mne.read_epochs(
                               self.data_path, 
                               preload=True, 
                               verbose=False
                               )
        ### Restricting data to EEG
        epochs = epochs.pick_types(eeg=True)

        ### Checking baseline correction is fine
        if not epochs.baseline:
            epochs.apply_baseline(baseline=(None, 0))
        else:
            assert len(epochs.baseline) ==2
            assert round(epochs.baseline[0], 1) == -0.1
            assert epochs.baseline[1] == 0.

        ### For decoding, considering only time points after 150 ms
        if args.analysis in ['whole_trial']:
            tmin = 0.3
            tmax = 0.8
            epochs.crop(tmin=tmin, tmax=tmax)
        else:
            tmin = -.1
            tmax = 1.2
        times = epochs.times
        all_times = times.copy()
        
        ### Transforming to numpy array
        epochs_array = epochs.get_data()
        assert epochs_array.shape[1] == 128

        ### Setting some time-frequency variables
        decimation = 1
        sampling_frequency = epochs.info['sfreq']
        frequencies = tfr_frequencies(args)
        ### Converting to / loading time frequency data
        if args.data_kind not in ['erp', 'D26','C22','C23', 'C11', 'B16']:
            n_cycles = frequencies / 2

            ### Transforming data into TFR using morlet wavelets
            print('Now transforming into time-frequency...')
            if 'phase' in args.data_kind:
                phase_or_power = 'phase'
            else:
                phase_or_power = 'power'
            
            epochs_array = mne.time_frequency.tfr_array_morlet(
                                            epochs_array, 
                                            sfreq=sampling_frequency,
                                            decim=decimation,
                                            freqs=frequencies, 
                                            n_cycles=n_cycles, 
                                            n_jobs=int(os.cpu_count()/9),
                                            output=phase_or_power,
                                            )
            ### Averaging the band
            epochs_array = numpy.average(epochs_array, axis=2)
            assert epochs_array.shape[1] == 128

            ### Decimating times
            if decimation > 1:
                times = times[::decimation]
                all_times = times.copy()[::decimation]
            ### Checking
            assert len(times) == epochs_array.shape[-1]

            ### Baseline correct the tfr data (subtracts the mean, takes the log (dB conversion))
            #tfr_epochs = mne.baseline.rescale(epochs_array, times, 
            #                                  baseline=(min(times), 0.),
            #                                  mode='logratio', 
            #                                  )

        ### Scaling 
        epochs_array = mne.decoding.Scaler(epochs.info, \
                    scalings='mean'\
                    ).fit_transform(epochs_array)
        ### Collecting the data shape
        data_shape = epochs_array.shape[1:]

        ### organizing to a dictionary having trigger as key
        ### numpy array or ERPs as values
        full_data_dict = collections.defaultdict(list)
        for i, e in enumerate(epochs.events[:, 2]):
            full_data_dict[int(e)].append(epochs_array[i])
        full_data_dict = {k : numpy.array(v) for k, v in full_data_dict.items()}

        ### reducing temporal resolution
        reduced_times = numpy.array(range(int(-.1*1000), int(1.2*1000), args.temporal_resolution)) / 1000
        resolution_ms = args.temporal_resolution / 1000
        rolling_window = [[t_i for t_i, t in enumerate(times) if (t>=red_t and t < red_t+resolution_ms)] for red_t in reduced_times]
        for k, epo in full_data_dict.items():
            sub_list = list()
            rolled_average = numpy.array([numpy.average(epo[:, :, window[0]:window[1]+1], axis=2) if len(window)>1 else epo[:, :, window[0]] for window in rolling_window])
            sub_epo = numpy.moveaxis(rolled_average, 0, 2)
            full_data_dict[k] = sub_epo
        ### Updating times
        times = reduced_times + resolution_ms/2
        #times = [times[i+int(sub_amount/2)] for i in sub_indices]

        '''
        ### Subsampling by average by sub_amount
        if 'subsample' in args.subsample:
            sub_amount = int(args.subsample.split('_')[-1])
            sub_indices = list(range(len(times)))[::sub_amount][:-1]
            for k, epo in full_data_dict.items():
                sub_list = list()
                rolled_average = numpy.array([numpy.average(epo[:, :, i:i+sub_amount], axis=2) for i in sub_indices])
                sub_epo = numpy.moveaxis(rolled_average, 0, 2)
                full_data_dict[k] = sub_epo
            ### Updating times
            times = [times[i+int(sub_amount/2)] for i in sub_indices]
        '''

        ### Reducing the number of ERPs by averaging, if required

        # 0: takes the min-max number of ERPs present across all stimuli
        min_value_to_use = min([len(v) for k, v in full_data_dict.items()])
        stimuli_values = {k : len(v) for k, v in full_data_dict.items()}
        print(
                'subject {} - Min-max amount of available ERPs: {} - median&std of max across stimuli: {}, {}'.format(
                       self.subject,
                       min_value_to_use, 
                       numpy.median(list(stimuli_values.values())), 
                       numpy.std(list(stimuli_values.values())))
             )
        ### leaving n=8 out for training
        leave_out_idxs = [1, 3, 5, 7, 9, 11, 13, 15]
        ### vector selection
        if args.average == -12:
            data_dict = {k : numpy.average(v[1::2], axis=0) for k, v in full_data_dict.items()}
        ## train/test
        elif args.average == -36:
            data_dict = {k : numpy.average(v[::2], axis=0) for k, v in full_data_dict.items()}
        elif args.average == -8:
            data_dict = {k : numpy.average(v[leave_out_idxs], axis=0) for k, v in full_data_dict.items()}
        elif args.average == -16:
            data_dict = {k : numpy.average(v[[i for i in range(len(v)) if i not in leave_out_idxs]], axis=0) for k, v in full_data_dict.items()}
        else:
            ### Using min-max of ERPs across stimuli
            if args.average == 0:
                stimuli_values = {k : min_value_to_use for k in stimuli_values.keys()}
            ### Using a fixed amount of ERPs
            else:
                stimuli_values = {k : min(v, args.average) for k, v in stimuli_values.items()}
            data_dict = dict()
            for k, v in full_data_dict.items():
                ### averaging
                averaged_v = numpy.average(
                                           random.sample(v.tolist(), 
                                           k=stimuli_values[k]), 
                                           axis=0
                                           )
                assert averaged_v.shape == v[0].shape
                data_dict[k] = averaged_v

        ### Reducing EEG to trig_to_info keys
        full_data_dict = {k : v for k, v in full_data_dict.items() if k in self.experiment.trigger_to_info.keys()}
        data_dict = {k : v for k, v in data_dict.items() if k in self.experiment.trigger_to_info.keys()}

        return full_data_dict, data_dict, times, all_times, frequencies, data_shape
