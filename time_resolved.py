import numpy
import os
import random
import scipy

from scipy import spatial, stats
from skbold.preproc import ConfoundRegressor
from tqdm import tqdm

from general_utils import evaluate_pairwise, prepare_file, prepare_folder, evaluation_round
from io_utils import ExperimentInfo, LoadEEG
from read_word_vectors import load_vectors

def time_resolved(all_args):

    args = all_args[0]
    n = all_args[1]
    searchlight = all_args[2]
    searchlight_clusters = all_args[3]

    out_path = prepare_folder(args)
    out_file, language_agnostic = prepare_file(args, n)

    file_path = os.path.join(out_path, out_file)

    ### Loading the experiment
    experiment = ExperimentInfo(args, subject=n)
    ### Loading the EEG data
    all_eeg = LoadEEG(args, experiment, n)
    eeg = all_eeg.data_dict
    if args.data_kind in ['D26', 'B16', 'C23']:
        indices = searchlight_clusters.neighbors[searchlight_clusters.code_to_index[args.data_kind]]
        eeg = {k : v[indices, :] for k, v in eeg.items()}
        
    comp_vectors = load_vectors(args, experiment, n)
    eeg = {experiment.trigger_to_info[k][0] : v for k, v in eeg.items()}

    ### removing instances of test sets having the same values
    if list(set([type(v) for v in comp_vectors.values()]))[0] in [int, float, numpy.float64]:
        clean_splits = list()
        for test_split in experiment.test_splits:
            if comp_vectors[experiment.trigger_to_info[test_split[0]][0]] == comp_vectors[experiment.trigger_to_info[test_split[1]][0]]:
                continue
            else:
                clean_splits.append(test_split)
        experiment.test_splits = clean_splits

    ### Words
    stimuli = list(eeg.keys())
    if args.experiment_id == 'two' and args.input_target_model == 'ceiling':
        stimuli = [s for s in stimuli if s in comp_vectors.keys()]
    for s in stimuli:
        #print(s)
        assert s in comp_vectors.keys()
    ### splitting into batches of 2 to control word length

    if not searchlight:
        sub_scores = list()
        for t in tqdm(range(len(all_eeg.times))):
            current_eeg = {k : v[:, t] for k, v in eeg.items()}

            corr = evaluation_round(args, experiment, current_eeg, comp_vectors)
            sub_scores.append(corr)

        with open(os.path.join(file_path), 'w') as o:
            for t in all_eeg.times:
                o.write('{}\t'.format(t))
            o.write('\n')
            for d in sub_scores:
                o.write('{}\t'.format(d))
    else:
        results_dict = dict()
        tmin = -.1
        tmax = 1.2
        relevant_times = list(range(int(tmin*10000), int(tmax*10000), searchlight_clusters.time_radius))
        out_times = list()

        electrode_indices = [searchlight_clusters.neighbors[center] for center in range(128)]
        for places in electrode_indices:
            for time in relevant_times:
                start_time = min([t_i for t_i, t in enumerate(all_eeg.times) if t>(time/10000)])
                out_times.append(start_time)
                end_time = max([t_i for t_i, t in enumerate(all_eeg.times) if t<=(time+searchlight_clusters.time_radius)/10000])+1
                #print([start_time, end_time])
                current_eeg = {k : v[places, start_time:end_time].flatten() for k, v in eeg.items()}
                corr = evaluation_round(args, experiment, current_eeg, comp_vectors)
                results_dict[(places[0], start_time)] = corr
        out_times = sorted(set(out_times))

        results_array = list()
        for e in range(128):
            e_row = list()
            for time in out_times:
                e_row.append(results_dict[(e, time)])
            results_array.append(e_row)

        results_array = numpy.array(results_array)

        with open(file_path, 'w') as o:
            for t in out_times:
                t = all_eeg.times[t]+(searchlight_clusters.time_radius/20000)
                o.write('{}\t'.format(t))
            o.write('\n')
            for e in results_array:
                for t in e:
                    o.write('{}\t'.format(t))
                o.write('\n')
