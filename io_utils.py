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
        self.events_log, self.trigger_to_info, self.response_times = self.read_events_log()
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
        ### loading response times

        response_times = {s : dict() for s in range(1, self.subjects+1)}
        for sub, stim, t in zip(
                                full_log['subject'], 
                                full_log['trial_type'], 
                                full_log['response_time']
                                ):
            if t != 'na':
                if stim not in response_times[int(sub)].keys():
                    response_times[int(sub)][stim] = [float(t)]
                else:
                    response_times[int(sub)][stim].append(float(t))

        return full_log, trig_to_info, response_times

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
        ### removing useless leave-two-out #all_combs = [c for c in all_combs if c[0] in relevant_trigs and c[1] in relevant_trigs
        return all_combs

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

        ### vector selection

        ### comp vectors for ceiling
        if args.average == -12:
            data_dict = {k : numpy.average(v[1::2], axis=0) for k, v in full_data_dict.items()}
        ### eeg data
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
