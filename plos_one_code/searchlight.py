import collections
import itertools
import numpy
import os
import scipy

from scipy import stats
from tqdm import tqdm
from general_utils import prepare_folder
from time_resolved import evaluation_round, prepare_data

class SearchlightClusters:

    def __init__(self, args):

        ### space 
        if args.searchlight_spatial_radius == 'large_distance':
            self.max_distance = 0.03
        elif args.searchlight_spatial_radius == 'small_distance':
            self.max_distance = 0.02
        elif args.searchlight_spatial_radius == 'fixed':
            self.max_distance = 5
        ### time
        if args.searchlight_temporal_radius == 'large':
            self.time_radius = 1000
        if args.searchlight_temporal_radius == 'medium':
            self.time_radius = 500
        if args.searchlight_temporal_radius == 'small':
            self.time_radius = 250

        ### clusters
        self.index_to_code = self.indices_to_codes()
        self.code_to_index = {v : k for k,v in self.index_to_code.items()}
        self.neighbors = self.read_searchlight_clusters()
        self.mne_adjacency_matrix = self.create_adjacency_matrix()

        self.tmin = -.1
        self.tmax = 1.2
        self.time_step = 10000
        self.relevant_times = list(range(int(self.tmin*self.time_step), int(self.tmax*self.time_step), self.time_radius))

    def create_adjacency_matrix(self):
        data = list()
        indices = list()
        index_pointer = [0]
        for i, kv in enumerate(self.neighbors.items()):
            v = kv[1][1:]
            for neighbor in v:
                indices.append(int(neighbor))
                data.append(1)
            index_pointer.append(len(indices))

        ### Just checking everything went fine
        mne_sparse_adj_matrix = scipy.sparse.csr_matrix((data, indices, index_pointer), dtype=int)
        for ikv, kv in enumerate(self.neighbors.items()):
            v = kv[1][1:]

            assert sorted([i for i, k in enumerate(mne_sparse_adj_matrix.toarray()[ikv]) if k == 1]) == sorted(v)

        return mne_sparse_adj_matrix 

    def indices_to_codes(self):

        index_to_code = collections.defaultdict(str)
        with open('searchlight/searchlight_clusters_{}mm.txt'.format(float(self.max_distance*1000)), 'r') as searchlight_file:
            for l in searchlight_file:
                if 'CE' not in l:
                    l = l.strip().split('\t')
                    index_to_code[int(l[1])] = l[0]

        return index_to_code

    def read_searchlight_clusters(self):

        searchlight_clusters = collections.defaultdict(list)

        with open('searchlight/searchlight_clusters_{}mm.txt'.format(float(self.max_distance*1000)), 'r') as searchlight_file:
            print(searchlight_file)
            for l in searchlight_file:
                if 'CE' not in l:
                    l = [int(i) for i in l.strip().split('\t')[1:]]
                    searchlight_clusters[l[0]] = l

        print('Average number of electrodes per cluster: {}'.format(
                round(
                      numpy.average([len(v) for v in searchlight_clusters.values()]), 
                      1
                      )
                )
                )

        return searchlight_clusters

def searchlight(all_args):

    args = all_args[0]
    all_eeg = all_args[1]
    comp_vectors = all_args[2]
    eeg = all_args[3]
    experiment = all_args[4]
    places = all_args[5][0]
    time = all_args[5][1]
    searchlight_clusters = all_args[6]

    start_time = min([t_i for t_i, t in enumerate(all_eeg.times) if t>(time/searchlight_clusters.time_step)])
    end_time = max([t_i for t_i, t in enumerate(all_eeg.times) if t<=(time+searchlight_clusters.time_radius)/searchlight_clusters.time_step])+1
    #print([start_time, end_time])
    current_eeg = {k : v[places, start_time:end_time].flatten() for k, v in eeg.items()}
    corr = evaluation_round(args, experiment, current_eeg, comp_vectors)

    #results_dict[(places[0], start_time)] = corr
    return places[0], start_time, corr

def searchlight_two(all_args):

    args = all_args[0]
    #all_eeg = all_args[1]
    #comp_vectors = all_args[2]
    #eeg = all_args[3]
    experiment = all_args[1]
    #places = all_args[5][0]
    #time = all_args[5][1]
    n = all_args[2]
    searchlight_clusters = all_args[3]
    places_and_times = all_args[4]

    all_eeg, comp_vectors, eeg, experiment, file_path = prepare_data((args, n))
    if args.input_target_model == 'ceiling':
        full_comp_vectors = comp_vectors.copy()
    results_dict = dict()
    for place_time in tqdm(places_and_times):
        places = place_time[0]
        time = place_time[1]

        start_time = min([t_i for t_i, t in enumerate(all_eeg.times) if t>(time/searchlight_clusters.time_step)])
        end_time = max([t_i for t_i, t in enumerate(all_eeg.times) if t<=(time+searchlight_clusters.time_radius)/searchlight_clusters.time_step])+1
        #print([start_time, end_time])
        current_eeg = {k : v[places, start_time:end_time].flatten() for k, v in eeg.items()}
        if args.input_target_model == 'ceiling':
            comp_vectors = {k : v[places, start_time:end_time].flatten() for k, v in full_comp_vectors.items()}
        corr = evaluation_round(args, experiment, current_eeg, comp_vectors)

        results_dict[(places[0], start_time)] = corr
    write_searchlight(all_eeg, file_path, results_dict, searchlight_clusters)
    #return places[0], start_time, corr

def write_searchlight(all_eeg, file_path, results_dict, searchlight_clusters):

    out_times = [k[1] for k in results_dict.keys()]
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
            t = all_eeg.times[t]+(searchlight_clusters.time_radius/(searchlight_clusters.time_step*2))
            o.write('{}\t'.format(t))
        o.write('\n')
        for e in results_array:
            for t in e:
                o.write('{}\t'.format(t))
            o.write('\n')
