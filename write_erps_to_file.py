import numpy
import os

from general_utils import read_args
from io_utils import ExperimentInfo, LoadEEG

args = read_args()

#if args.average != -12:
#    raise RuntimeError('average must be -12!')
if args.temporal_resolution != 10:
    raise RuntimeError('time resolution should be 10ms!')
if args.experiment_id == 'one':
    if args.semantic_category_two != 'individual':
        raise RuntimeError('also extracting vectors for categories, which is not supported!')
if args.experiment_id == 'two':
    if args.semantic_category_two != 'famous':
        raise RuntimeError('at the moment, only extraction for famous entities is supported!')

output_folder = os.path.join(
                             '/', 'import', 'cogsci', 'andrea', 
                             'github', 'entity_central', 'brain_data', 
                             args.experiment_id
                             )
os.makedirs(output_folder, exist_ok=True)

times_and_labels = [
                    ((0., .2), '0-200ms'),
                    ((.2, .3), '200-300ms'),
                    ((.3, .5), '300-500ms'),
                    ((.5, .8), '500-800ms'),
                    ]

for times, times_label in times_and_labels:

    with open(os.path.join(
                           output_folder, 
                           'rough_and_ready_exp_{}_average_{}_{}.eeg'.format(args.experiment_id, args.average, times_label)
                           ), 'w') as o:
        o.write('subject\tstimulus\teeg_vector\n')
        for sub in range(1, 34):
            experiment = ExperimentInfo(
                                        args, 
                                        subject=sub
                                        )
            eeg_data = LoadEEG(args, experiment, sub)
            minimum_time = times[0]
            max_time = times[1]
            rel_indices = [t_i for t_i, t in enumerate(eeg_data.times) if t>minimum_time and t<max_time]
            eeg = {experiment.trigger_to_info[k][0] : v[:, rel_indices].flatten() for k, v in eeg_data.data_dict.items()}
            for k, v in eeg.items():
                o.write('{}\t{}\t'.format(sub, k))
                for dim in v:
                    o.write('{}\t'.format(dim))
                o.write('\n')
