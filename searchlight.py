import collections
import numpy
import os
import scipy

from scipy import stats
from tqdm import tqdm
from general_utils import prepare_folder

def write_plot_searchlight(args, n, explicit_times, results_array):

    output_folder = prepare_folder(args)

    if 'classification' in args.analysis:
        input_file = os.path.join(output_folder, 
                  'sub-{:02}.rsa'.format(n))
    else:
        input_file = os.path.join(output_folder,
                  '{}_sub-{:02}.rsa'.format(args.input_target_model, n))

    ### adding information about cluster_size
    input_file = input_file.replace('.rsa', '_spatial_{}_temporal_{}.rsa'.format(args.searchlight_spatial_radius, args.searchlight_temporal_radius))

    with open(input_file, 'w') as o:
        for t in explicit_times:
            o.write('{}\t'.format(t))
        o.write('\n')
        for e in results_array:
            for t in e:
                o.write('{}\t'.format(t))
            o.write('\n')

def join_searchlight_results(results, relevant_times):

    results_array = list()
    results_dict = {r[0] : r[1] for r in results}

    for e in range(128):
        e_row = list()
        for t in relevant_times:
            e_row.append(results_dict[(e, t)])
        results_array.append(e_row)

    results_array = numpy.array(results_array)

    return results_array

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

        return searchlight_clusters
