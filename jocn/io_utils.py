import os
import mne
import random
import re
import collections
import numpy
import itertools
import scipy

from mne import stats
from scipy import stats
from tqdm import tqdm

#from lab.utils import read_words, read_trigger_ids, select_words
#from searchlight import SearchlightClusters

def statistic(a, b):
    #sem = scipy.stats.sem(a.tolist()+b.tolist())
    sem = numpy.std(a.tolist()+b.tolist()) / numpy.sqrt(len(a.tolist())+len(b.tolist()))
    stat = numpy.mean(a) - numpy.mean(b)
    return stat / sem

class ExperimentInfo:

    def __init__(self, args, subject=1):
        
        self.analysis = args.analysis
        self.mapping_model = args.mapping_model
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

            fold = 'derivatives'
            sub_path = os.path.join(
                                    self.data_folder,
                                    fold,
                                    'sub-{:02}'.format(s),
                                    'sub-{:02}_task-namereadingimagery_eeg-epo.fif.gz'.format(s)
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
                key_one = 'semantic_domain'
                key_two = 'familiarity'
                cat_one = full_log[key_one][t_i]
                cat_two = full_log[key_two][t_i]
                infos = [name, cat_one, cat_two]
                t = int(t)
                if t in trig_to_info.keys():
                    assert trig_to_info[t] == infos
                trig_to_info[t] = infos
        ### loading response times

        response_times = {s : dict() for s in range(1, self.subjects+1)}
        cat_response_times = dict()
        general_response_times = dict()
        accuracies = dict()
        for sub, stim, t, acc, sem, fam in zip(
                                full_log['subject'], 
                                full_log['trial_type'], 
                                full_log['response_time'],
                                full_log['accuracy'],
                                full_log['semantic_domain'],
                                full_log['familiarity'],
                                ):
            if t != 'na':
                if stim not in response_times[int(sub)].keys():
                    response_times[int(sub)][stim] = [float(t)]
                else:
                    response_times[int(sub)][stim].append(float(t))
                if '{}_{}'.format(sem, fam) not in cat_response_times.keys():
                    cat_response_times['{}_{}'.format(sem, fam)] = [float(t)]
                    accuracies['{}_{}'.format(sem, fam)] = [float(acc)]
                else:
                    cat_response_times['{}_{}'.format(sem, fam)].append(float(t))
                    accuracies['{}_{}'.format(sem, fam)].append(float(acc))
                for val in [sem, fam]:
                    if val not in cat_response_times.keys():
                        cat_response_times[val] = [float(t)]
                        accuracies[val] = [float(acc)]
                    else:
                        cat_response_times[val].append(float(t))
                        accuracies[val].append(float(acc))
        os.makedirs('results', exist_ok=True)
        behav_file = os.path.join('results', 'behavioural_averages.txt')
        if not os.path.exists(behav_file):
            ### statistical comparisons
            print('now running statistical comparisons on response times...')
            comparisons = dict()
            for k_one, v_one in cat_response_times.items():
                for k_two, v_two in cat_response_times.items():
                    if k_one == k_two:
                        continue
                    if tuple(sorted([k_one, k_two])) in [tuple(sorted(v)) for v in comparisons.keys()]:
                        continue
                    ### regular
                    #comp = scipy.stats.ttest_ind(v_one, v_two)
                    comp = scipy.stats.permutation_test(
                                                        [numpy.array(v_one),numpy.array(v_two)], 
                                                        statistic=statistic,
                                                        n_resamples=1000,
                                                        )
                    ### log
                    log_comp = scipy.stats.permutation_test(
                                                            [numpy.log(v_one), numpy.log(v_two)], 
                                                            statistic=statistic,
                                                            n_resamples=1000,
                                                            )
                    #log_comp = scipy.stats.ttest_ind(numpy.log(v_one), numpy.log(v_two))
                    comparisons[(k_one, k_two)] = [comp.statistic, comp.pvalue, log_comp.statistic, log_comp.pvalue]
            ### regular
            ps = [v[1] for k, v in comparisons.items()]
            corr_ps = mne.stats.fdr_correction(ps)[1]
            for kv, p in zip(comparisons.items(), corr_ps):
                comparisons[kv[0]].append(p)
            ### log
            ps = [v[3] for k, v in comparisons.items()]
            corr_ps = mne.stats.fdr_correction(ps)[1]
            for kv, p in zip(comparisons.items(), corr_ps):
                comparisons[kv[0]].append(p)

            ### writing comparisons to file
            with open(behav_file, 'w') as o:
                o.write('accuracies\n\n')
                overall = [val for v in accuracies.values() for val in v]
                o.write('overall\t{}\n'.format(sum(overall)/len(overall)))
                for k, v in accuracies.items():
                    acc = sum(v) / len(v)
                    o.write('{}\t{}\n'.format(k, acc))
                o.write('\n\nresponse times\n\n')
                o.write('cat\traw_avg\traw_std\tlog_avg\tlog_std\n')
                for k, v in cat_response_times.items():
                    o.write('{}\t{}\t{}\t{}\t{}\n'.format(
                                                  k, 
                                                  numpy.average(v), 
                                                  numpy.std(v),
                                                  numpy.average(numpy.log(v)), 
                                                  numpy.std(numpy.log(v))
                                                  )
                                                  )
                o.write('response times comparisons\n\n')
                o.write('cats\traw t\toriginal p\tlog t\tlog original p\tcorrected p\tlog corrected p\n')
                for k, vals in comparisons.items():
                    o.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                                                                  k, 
                                                                  round(vals[0], 5), 
                                                                  round(vals[1], 5),
                                                                  round(vals[2], 5),
                                                                  round(vals[3], 5),
                                                                  round(vals[4], 5),
                                                                  round(vals[5], 5),
                                                                  ),
                                                                  )

        ### writing to file for R, if file is not there
        out_file = os.path.join('data', 'r_file.tsv')
        if not os.path.exists(out_file):
            ### setting the mapper
            mapper = {
                      'trial_type' : {k : len(k) for k in full_log['trial_type']},
                      'value' : {v: k for k, v in enumerate(sorted(list(set(full_log['value']))))},
                      'semantic_domain' : {'person' : -1, 'place' : 1}, 
                      'familiarity' : {'famous' : -1, 'familiar' : 1},
                      }
            keys = [v for v in list(full_log.keys()) if v not in ['onset', 'duration']]
            with open(out_file, 'w') as o:
                for w in keys:
                    if w == 'trial_type':
                        w = 'name_length'
                    o.write('{}\t'.format(w))
                o.write('\n')
                for i in range(len(full_log['value'])):
                    for k in keys:
                        val = full_log[k][i]
                        if k in mapper.keys():
                            val = mapper[k][val]
                        o.write('{}\t'.format(val))
                    o.write('\n')


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
            relevant_trigs_two = [k for k, v in self.trigger_to_info.items() if v[2]==self.semantic_category_two]
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
        self.full_data, self.data_dict, \
        self.times, self.all_times, \
        self.frequencies, self.data_shape = self.load_epochs(args)

    def load_epochs(self, args):

        frequencies = dict()
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
        tmin = -.1
        tmax = 1.2
        times = epochs.times
        all_times = times.copy()
        
        ### Transforming to numpy array
        epochs_array = epochs.get_data()
        assert epochs_array.shape[1] == 128

        ### Scaling 
        epochs_array = mne.decoding.Scaler(
                                        epochs.info,
                                        #scalings='median',
                                        scalings='mean',
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
        temporal_resolution = 5
        reduced_times = numpy.array(range(int(-.1*1000), int(1.2*1000), temporal_resolution)) / 1000
        resolution_ms = temporal_resolution / 1000
        rolling_window = [[t_i for t_i, t in enumerate(times) if (t>=red_t and t < red_t+resolution_ms)] for red_t in reduced_times]
        for k, epo in full_data_dict.items():
            sub_list = list()
            rolled_average = numpy.array([numpy.average(epo[:, :, window[0]:window[1]+1], axis=2) if len(window)>1 else epo[:, :, window[0]] for window in rolling_window])
            sub_epo = numpy.moveaxis(rolled_average, 0, 2)
            full_data_dict[k] = sub_epo
        ### Updating times
        times = reduced_times + resolution_ms/2

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

        ### eeg data
        ### Using a fixed amount of ERPs
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
