import numpy
import os

from tqdm import tqdm

from general_utils import prepare_file, prepare_folder, evaluation_round
from io_utils import ExperimentInfo, LoadEEG
from read_word_vectors import load_vectors

def prepare_data(all_args):

    args = all_args[0]
    n = all_args[1]

    out_path = prepare_folder(args)
    out_file, language_agnostic = prepare_file(args, n)

    file_path = os.path.join(out_path, out_file)

    if args.across_subjects:
        eeg = dict()
        for across_n in range(1, n+1):
            ### Loading the experiment
            experiment = ExperimentInfo(args, subject=across_n)
            ### Loading the EEG data
            all_eeg = LoadEEG(args, experiment, across_n)
            sub_eeg = all_eeg.data_dict
            for k, v in sub_eeg.items():
                if k not in eeg.keys():
                    eeg[k] = v
                else:
                    eeg[k] = numpy.average([eeg[k], v], axis=0)
    else:
        ### Loading the experiment
        experiment = ExperimentInfo(args, subject=n)
        ### Loading the EEG data
        all_eeg = LoadEEG(args, experiment, n)
        eeg = all_eeg.data_dict

    comp_vectors = load_vectors(args, experiment, n)
    eeg = {experiment.trigger_to_info[k][0] : v for k, v in eeg.items()}

    ### Words
    stimuli = list(eeg.keys())
    marker = False
    ### cases where not all stimuli are available
    if args.experiment_id == 'two' and args.input_target_model == 'ceiling':
        marker = True
    if marker:
        stimuli = [s for s in stimuli if s in comp_vectors.keys()]
        eeg = {s : eeg[s] for s in stimuli}
    for s in stimuli:
        #print(s)
        assert s in comp_vectors.keys()

    return all_eeg, comp_vectors, eeg, experiment, file_path

def time_resolved(all_args):

    args = all_args[0]
    all_eeg, comp_vectors, eeg, experiment, file_path = prepare_data(all_args)

    if args.input_target_model == 'ceiling':
        full_comp_vectors = comp_vectors.copy()

    sub_scores = list()
    for t in tqdm(range(len(all_eeg.times))):
        current_eeg = {k : v[:, t] for k, v in eeg.items()}
        if args.input_target_model == 'ceiling':
            comp_vectors = {k : v[:, t] for k, v in full_comp_vectors.items()}

        corr = evaluation_round(args, experiment, current_eeg, comp_vectors)
        sub_scores.append(corr)

    with open(os.path.join(file_path), 'w') as o:
        for t in all_eeg.times:
            o.write('{}\t'.format(t))
        o.write('\n')
        for d in sub_scores:
            o.write('{}\t'.format(d))
