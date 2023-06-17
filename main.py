import argparse
import itertools
import logging
import multiprocessing
import numpy
import os
import random
import scipy
import sklearn

from tqdm import tqdm

from general_utils import read_args
from io_utils import ExperimentInfo, LoadEEG, tfr_frequencies

from plot_classification import plot_classification
#from plotting.plot_decoding_results_breakdown import plot_decoding_results_breakdown
#from plotting.plot_decoding_scores_comparison import plot_decoding_scores_comparison

from searchlight import searchlight, SearchlightClusters, write_searchlight
from group_searchlight import group_searchlight

from read_word_vectors import load_vectors

from time_resolved import prepare_data, time_resolved

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

#numpy.seterr(all='raise')

args = read_args()

searchlight_clusters = SearchlightClusters(args)
print('Average number of electrodes per cluster: {}'.format(
                round(
                      numpy.average([len(v) for v in searchlight_clusters.neighbors.values()]), 
                      1
                      )
                )
                )

### Plotting

if args.plot:
    if args.analysis == 'whole_trial':
        if args.experiment_id == 'one':
            plot_decoding_results_breakdown(args)
            plot_feature_selection_comparison(args)
            plot_decoding_scores_comparison(args)
        else:
            plot_decoding_results_breakdown(args)
        plot_feature_selection_comparison(args, experiment)
    elif args.analysis == 'time_resolved':
        plot_classification(args)
    elif args.analysis == 'searchlight':
        group_searchlight(args)

### Getting the results
else:
    #processes = 1

    experiment = ExperimentInfo(args)

    if __name__ == '__main__':

        if args.analysis == 'whole_trial':
            raise RuntimeError('to be implemented!')

        processes = int(os.cpu_count()/2)

        ### time resolved
        if args.analysis == 'time_resolved':
            if args.debugging:
                for n in range(1, experiment.subjects+1):
                    time_resolved([args, n, searchlight_marker, searchlight_clusters])
            else:
                with multiprocessing.Pool(processes=processes) as pool:
                    pool.map(time_resolved, [(args, n) for n in range(1, experiment.subjects+1)])
                    pool.close()
                    pool.join()

        ### searchlight
        elif args.analysis == 'searchlight':
            #processes = int(os.cpu_count()-2)
            for n in range(1, experiment.subjects+1):
                #searchlight([args, n, searchlight_marker, searchlight_clusters])

                all_eeg, comp_vectors, eeg, experiment, file_path = prepare_data((args, n))

                results_dict = dict()

                electrode_indices = [searchlight_clusters.neighbors[center] for center in range(128)]

                for places in tqdm(electrode_indices):
                    if args.debugging:
                        for time in searchlight_clusters.relevant_times:
                            res = searchlight((args, all_eeg, comp_vectors, eeg, experiment, places, time, searchlight_clusters)) 
                            results_dict[(res[0], res[1])] = res[2]
                    else:
                        with multiprocessing.Pool(processes=processes) as pool:
                            results_list = pool.map(searchlight, [(args, all_eeg, comp_vectors, eeg, experiment, places, t, searchlight_clusters) for t in searchlight_clusters.relevant_times])
                            pool.close()
                            pool.join()
                        for res in results_list:
                            results_dict[(res[0], res[1])] = res[2]

                write_searchlight(file_path, results_dict, searchlight_clusters)
