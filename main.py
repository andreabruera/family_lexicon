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

from general_utils import how_many_cores, plot_erps, plot_scalp_erps, read_args
from io_utils import ExperimentInfo, LoadEEG, tfr_frequencies

from plot_classification import plot_classification

from searchlight import searchlight, searchlight_two, SearchlightClusters, write_searchlight
from group_searchlight import group_searchlight

from read_word_vectors import load_vectors

from time_resolved import prepare_data, time_resolved

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

#numpy.seterr(all='raise')

args = read_args()

searchlight_clusters = SearchlightClusters(args)

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

### Running the analyses
else:

    experiment = ExperimentInfo(args)

    ### plotting erps
    plot_erps(args)
    #plot_scalp_erps(args, searchlight_clusters)
    import pdb; pdb.set_trace()

    if __name__ == '__main__':

        if args.analysis == 'whole_trial':
            raise RuntimeError('to be implemented!')

        processes = how_many_cores(args)
        #processes = 2

        ### time resolved
        if args.analysis == 'time_resolved':

            if args.across_subjects:
                time_resolved([args, experiment.subjects])
            #else:
            #    continue

            if args.debugging:
                for n in range(1, experiment.subjects+1):
                    time_resolved([args, n])
            else:
                with multiprocessing.Pool(processes=processes) as pool:
                    pool.map(time_resolved, [(args, n) for n in range(1, experiment.subjects+1)])
                    pool.close()
                    pool.join()

        ### searchlight
        elif args.analysis == 'searchlight':

            electrode_indices = [searchlight_clusters.neighbors[center] for center in range(128)]

            places_and_times = list(itertools.product(electrode_indices, searchlight_clusters.relevant_times))

            if args.debugging:
                for n in tqdm(range(1, experiment.subjects+1)):
                    res = searchlight_two((args, experiment, n, searchlight_clusters, places_and_times)) 
            else:
                with multiprocessing.Pool(processes=processes) as pool:
                    pool.map(searchlight_two, [(args, experiment, n, searchlight_clusters, places_and_times) for n in range(1, experiment.subjects+1)])

            '''
            electrode_indices = [searchlight_clusters.neighbors[center] for center in range(128)]

            places_and_times = list(itertools.product(electrode_indices, searchlight_clusters.relevant_times))

            for n in tqdm(range(1, experiment.subjects+1)):

                all_eeg, comp_vectors, eeg, experiment, file_path = prepare_data((args, n))

                results_dict = dict()

                ### each place_time separately
                if args.debugging:
                    for place_time in places_and_times:
                        res = searchlight((args, all_eeg, comp_vectors, eeg, experiment, place_time, searchlight_clusters)) 
                        results_dict[(res[0], res[1])] = res[2]

                ### multiprocessing within one subject
                else:
                    #processes = 2
                    with multiprocessing.Pool(processes=processes) as pool:
                        results_list = pool.map(searchlight, [(args, all_eeg, comp_vectors, eeg, experiment, place_time, searchlight_clusters) for place_time in places_and_times])
                        pool.close()
                        pool.join()
                    ### reordering results
                    for res in results_list:
                        results_dict[(res[0], res[1])] = res[2]

                ### writing to files
                write_searchlight(all_eeg, file_path, results_dict, searchlight_clusters)
            '''
