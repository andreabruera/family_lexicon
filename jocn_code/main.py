import itertools
import logging
import multiprocessing

from tqdm import tqdm

from general_utils import how_many_cores, plot_erps, plot_scalp_erps, read_args
from io_utils import ExperimentInfo, LoadEEG

from plot_classification import plot_classification

from searchlight import searchlight, searchlight_two, SearchlightClusters, write_searchlight
from group_searchlight import group_searchlight

from read_word_vectors import load_vectors

from time_resolved import prepare_data, time_resolved

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

args = read_args()

searchlight_clusters = SearchlightClusters(args)

### Plotting

if args.plot:
    if args.analysis == 'time_resolved':
        plot_classification(args)
    elif args.analysis == 'searchlight':
        group_searchlight(args)

### Running the analyses
else:

    experiment = ExperimentInfo(args)

    ### plotting erps
    plot_erps(args)
    plot_scalp_erps(args, searchlight_clusters)

    if __name__ == '__main__':

        processes = how_many_cores(args)

        ### time resolved
        if args.analysis == 'time_resolved':

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
