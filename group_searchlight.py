import argparse
import os
import collections
import numpy
import re
import logging
import itertools
import functools
import mne
import pickle
import scipy
import multiprocessing

from scipy import stats
from matplotlib import pyplot
from tqdm import tqdm

from general_utils import prepare_file, prepare_folder, return_baseline
from io_utils import ExperimentInfo, LoadEEG
from searchlight import SearchlightClusters

def group_searchlight(args):

    plot_path = prepare_folder(args).replace('results', 'plots')

    ### comparing models
    if args.comparison:
        if args.input_target_model not in ['famous_familiar', 'coarse_category']:
            models = ['xlm-roberta-large_individuals', args.input_target_model]
        elif args.input_target_model == 'famous_familiar':
            models = ['person', 'place']
        elif args.input_target_model == 'coarse_category':
            models = ['familiar', 'famous']
        collector = dict()
        for m in models:
            if args.input_target_model not in ['famous_familiar', 'coarse_category']:
                args.input_target_model = m
            elif args.input_target_model == 'famous_familiar':
                args.semantic_category_one = m
            elif args.input_target_model == 'coarse_category':
                args.semantic_category_two = m
            #significance = .005
            #significance = 0.1
            clusters = SearchlightClusters(args)
            electrode_index_to_code = clusters.index_to_code
            mne_adj_matrix = clusters.mne_adjacency_matrix
            all_subjects = list()
            input_folder = prepare_folder(args)

            for n in range(1, 34):
                out_file, language_agnostic = prepare_file(args, n)

                input_file = os.path.join(input_folder, out_file)

                print(input_file)
                assert os.path.exists(input_file)
                with open(input_file, 'r') as i:
                    lines = [l.strip().split('\t') for l in i.readlines()]
                ### reading times and electrodes
                times = numpy.array([w for w in lines[0]], dtype=numpy.float64)
                electrodes = numpy.array([[float(v) for v in l] for l in lines[1:]]).T
                assert electrodes.shape == (len(times), 128)
                ### reducing relevant time points
                ### following leonardelli & fairhall, range is 100-750ms

                #upper_limit = 0.8 if args.experiment_id == 'two' else 1.2
                upper_limit = 1.2 if args.experiment_id == 'two' else 1.2
                lower_limit = 0.0 if args.experiment_id == 'two' else 0.
                #lower_limit = 0.3 if args.experiment_id == 'two' else .3
                relevant_indices = [t_i for t_i, t in enumerate(times) if (t>lower_limit and t<upper_limit)]
                #relevant_indices = [t_i for t_i, t in enumerate(times) if t>0.]
                times = times[relevant_indices]
                electrodes = electrodes[relevant_indices, :]

                all_subjects.append(electrodes)

            random_baseline = return_baseline(args) 
            all_subjects = numpy.array(all_subjects) - random_baseline
            collector[m] = numpy.ones(shape=all_subjects.shape) - all_subjects
        if args.input_target_model not in ['famous_familiar', 'coarse_category']:
            all_subjects = collector[m] - collector['xlm-roberta-large_individuals']
        elif args.input_target_model == 'famous_familiar':
            all_subjects = collector[m] - collector['person']
            args.semantic_category_one = 'all'
        elif args.input_target_model == 'coarse_category':
            all_subjects = collector[m] - collector['familiar']
            args.semantic_category_two = 'all'

    ### simply plotting one model
    else:

        print(args.analysis)
        pyplot.rcParams['figure.constrained_layout.use'] = True

        plot_path = prepare_folder(args).replace('results', 'plots')
        clusters = SearchlightClusters(args)
        electrode_index_to_code = clusters.index_to_code
        mne_adj_matrix = clusters.mne_adjacency_matrix
        all_subjects = list()
        input_folder = prepare_folder(args)

        for n in range(1, 34):
            out_file, language_agnostic = prepare_file(args, n)

            input_file = os.path.join(input_folder, out_file)

            assert os.path.exists(input_file)
            with open(input_file, 'r') as i:
                lines = [l.strip().split('\t') for l in i.readlines()]
            ### reading times and electrodes
            times = numpy.array([w for w in lines[0]], dtype=numpy.float64)
            electrodes = numpy.array([[float(v) for v in l] for l in lines[1:]]).T
            assert electrodes.shape == (len(times), 128)
            ### reducing relevant time points
            ### following leonardelli & fairhall, range is 100-750ms

            upper_limit = 0.8 if args.experiment_id == 'two' else 1.2
            lower_limit = 0.0 if args.experiment_id == 'two' else 0.
            relevant_indices = [t_i for t_i, t in enumerate(times) if (t>lower_limit and t<upper_limit)]
            times = times[relevant_indices]
            electrodes = electrodes[relevant_indices, :]

            all_subjects.append(electrodes)

        random_baseline = return_baseline(args) 
        all_subjects = numpy.array(all_subjects) - random_baseline
    print(max(all_subjects.flatten()))

    t_stats, _, \
    p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(all_subjects, \
                                                       tail=1, \
                                                       adjacency=mne_adj_matrix, \
                                                       threshold=dict(start=0, step=0.2), \
                                                       n_jobs=os.cpu_count()-1, \
                                                       )
    logging.info('Minimum p-value for {}: {}'.format(args.input_target_model, min(p_values)))

    significance = .05
    original_shape = t_stats.shape
    avged_subjects = numpy.average(all_subjects, axis=0)
    assert avged_subjects.shape == original_shape

    for significance in [0.05, #0.1
                        ]:

        reshaped_p = p_values.copy()
        reshaped_p[reshaped_p>=significance] = 1.0
        reshaped_p = reshaped_p.reshape(original_shape).T

        ### relevant time indices
        significant_times = list()
        for t in range(reshaped_p.shape[-1]):
            if numpy.sum(reshaped_p[:, t]) < reshaped_p.shape[0]:
                significant_times.append(times[t])
        reshaped_p = p_values.copy()
        reshaped_p = reshaped_p.reshape(original_shape).T

        print(significant_times)

        #relevant_times
        tmin = times[0]
        if args.data_kind != 'time_frequency':
            if args.searchlight_temporal_radius == 'large':
                sfreq = 10
            elif args.searchlight_temporal_radius == 'medium':
                sfreq = 20 
            elif args.searchlight_temporal_radius == 'small':
                sfreq = 40

        elif args.data_kind == 'time_frequency':
            sfreq = 256 / 16
        info = mne.create_info(
                               ch_names=[v for k, v in clusters.index_to_code.items()],
                               sfreq=sfreq,
                               ch_types='eeg',
                               )

        #evoked = mne.EvokedArray(reshaped_p, info=info, tmin=tmin)
        evoked = mne.EvokedArray(avged_subjects.T, info=info, tmin=tmin)

        montage = mne.channels.make_standard_montage('biosemi128')
        evoked.set_montage(montage)

        os.makedirs(plot_path, exist_ok=True)

        ### Writing to txt
        channels = evoked.ch_names
        assert isinstance(channels, list)
        assert len(channels) == reshaped_p.shape[0]
        assert len(times) == reshaped_p.shape[-1]
        txt_path = os.path.join(plot_path, '{}_searchlight_rsa_significant_points_{}.txt'.format(args.input_target_model, significance))

        txt_path = txt_path.replace('.txt', '_spatial_{}_temporal_{}.txt'.format(
                                                   args.searchlight_spatial_radius,
                                                   args.searchlight_temporal_radius,
                                                   ))
        if not language_agnostic:
            txt_path = txt_path.replace(args.input_target_model, '{}_{}'.format(args.input_target_model, args.language))
        with open(txt_path, 'w') as o:
            o.write('Time\tElectrode\tp-value\tt-value\n')
            for t_i in range(reshaped_p.shape[-1]):
                time = times[t_i]
                for c_i in range(reshaped_p.shape[0]):
                    channel = channels[c_i]
                    p = reshaped_p[c_i, t_i]
                    p_value = reshaped_p[c_i, t_i]
                    t_value = t_stats.T[c_i, t_i]
                    o.write('{}\t{}\t{}\t{}\n'.format(time, channel, p_value, t_value))

        correction = 'corrected' if args.corrected else 'uncorrected'
        title='Searchlight for {} - {}'.format(args.input_target_model, args.semantic_category_one)
        title = '{} - {}, p<={}'.format(title, correction, significance)

        if args.evaluation_method == 'correlation':
            if args.semantic_category_one == args.semantic_category_two:
                if args.comparison:
                    vmax = 0.05
                else:
                    vmax = 0.1
            else:
                if args.comparison:
                    vmax = 0.05
                else:
                    vmax = 0.12
        else:
            if args.comparison:
                vmax = 0.1
            else:
                vmax = 0.2

        evoked.plot_topomap(ch_type='eeg', 
                            time_unit='s', 
                            times=evoked.times,
                            ncols='auto',
                            nrows='auto', 
                            vmax=vmax,
                            vmin=0.,
                            scalings={'eeg':1.}, 
                            cmap='Spectral_r',
                            mask=reshaped_p<=significance,
                            mask_params=dict(marker='o', markerfacecolor='black', markeredgecolor='black',
                                        linewidth=0, markersize=4),
                            #colorbar=False,
                            size = 3.,
                            title=title,
                            )

        ### building the file name
        f_name = '{}_{}_{}_spatial_{}_temporal_{}_{}.jpg'.format(
                                      args.input_target_model, 
                                      args.semantic_category_one, 
                                      args.semantic_category_two,
                                      args.searchlight_spatial_radius,
                                      args.searchlight_temporal_radius,
                                      correction, 
                                      )

        if not language_agnostic:
            f_name = f_name.replace(args.input_target_model, '{}_{}'.format(args.input_target_model, args.language))
        if args.comparison:
            f_name = f_name.replace('.jpg', '_comparison.jpg')
        print(f_name)
        f_name = os.path.join(plot_path, f_name)
        pyplot.savefig(f_name, dpi=600)
        pyplot.savefig(f_name.replace('jpg', 'svg'), dpi=600)
        pyplot.clf()
