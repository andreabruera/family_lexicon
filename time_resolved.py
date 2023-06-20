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

def prepare_data(all_args):

    args = all_args[0]
    n = all_args[1]

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

    return all_eeg, comp_vectors, eeg, experiment, file_path

def time_resolved(all_args):

    args = all_args[0]
    all_eeg, comp_vectors, eeg, experiment, file_path = prepare_data(all_args)

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
