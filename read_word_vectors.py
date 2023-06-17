import gensim
import itertools
import math
import numpy
import os
import pickle
import re
import scipy
import sklearn
import syllables

from gensim.models import KeyedVectors, Word2Vec
from qwikidata.linked_data_interface import get_entity_dict_from_api
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from tqdm import tqdm
from wikipedia2vec import Wikipedia2Vec

def purity(preds, labels):

    counter = {l : list() for l in list(set(labels))}
    for p, l in zip(preds, labels):
        counter[l].append(p)
    mapper = {m_k : {p : len([v for v in m_v if v == p]) for p in list(set(m_v))} for m_k, m_v in counter.items()}
    mapper = {k : max(v.items(), key=lambda item : item[1])[0] for k, v in mapper.items()}

    scores = 0
    for p, l in zip(preds, labels):
        if p == mapper[l]:
            scores += 1
    purity_score = scores / len(labels)

    return purity_score  

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = numpy.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    #print (matrix)
    return (matrix[size_x - 1, size_y - 1])

def load_vectors(args, experiment, n, clustered=False):

    names = [v[0] for v in experiment.trigger_to_info.values()]
    if args.input_target_model == 'ceiling':
        ceiling = dict()
        for n_ceiling in range(1, 34):
            eeg_data_ceiling = LoadEEG(args, experiment, n_ceiling)
            eeg_ceiling = eeg_data_ceiling.data_dict
            for k, v in eeg_ceiling.items():
                ### Adding and flattening
                if k not in ceiling.keys():
                    ceiling[k] = list()
                ceiling[k].append(v.flatten())
        comp_vectors = {k : numpy.average([vec for vec_i, vec in enumerate(v) if vec_i!=n-1], axis=0) for k, v in ceiling.items()}
        comp_vectors = {experiment.trigger_to_info[k][0] : v for k, v in comp_vectors.items()}

    elif args.input_target_model in ['coarse_category', 'famous_familiar', 'fine_category']:
        if args.input_target_model == 'coarse_category':
            categories = {v[0] : v[1] for v in experiment.trigger_to_info.values()}
            sorted_categories = sorted(set(categories.values()))
        elif args.input_target_model == 'famous_familiar':
            if args.experiment_id == 'one':
                raise RuntimeError('There is no famous_familiar distinction for this experiment!')
            categories = {v[0] : v[2] for v in experiment.trigger_to_info.values()}
            sorted_categories = sorted(set(categories.values()))
        elif args.input_target_model == 'fine_category':
            if args.experiment_id == 'two':
                raise RuntimeError('There is no famous_familiar distinction for this experiment!')
            categories = {v[0] : v[2] for v in experiment.trigger_to_info.values()}
            people_categories = {v[0] : v[2] for v in experiment.trigger_to_info.values() if v[1]=='person'}
            sorted_people_categories = sorted(set(people_categories.values()))
            place_categories = {v[0] : v[2] for v in experiment.trigger_to_info.values() if v[1]=='place'}
            sorted_place_categories = sorted(set(place_categories.values()))
            sorted_categories = sorted_people_categories + sorted_place_categories
        #vectors = {k_one : numpy.array([0. if categories[k_one]==categories[k_two] else 1. for k_two in names if k_one != k_two]) for k_one in names}
        vectors = {w : sorted_categories.index(categories[w]) for w in names}
        vectors = minus_one_one_norm(vectors.items())

    elif args.input_target_model == 'individuals':
        if args.experiment_id == 'one':
            sorted_infos = [experiment.trigger_to_info[k] for k in sorted(experiment.trigger_to_info.keys())]
            people = [v[0] for v in sorted_infos if v[1]=='person']
            places = [v[0] for v in sorted_infos if v[1]=='place']
            ordered_names = people + places
            vectors = {k : k_i for k_i, k in enumerate(ordered_names)}
        else:
            coarse = {v[0] : v[1] for v in experiment.trigger_to_info.values()}
            fame = {v[0] : v[2] for v in experiment.trigger_to_info.values()}
            ordered_names = [co[0] for co, fa in zip(coarse.items(), fame.items()) if co[1]=='person' and fa[1]=='familiar'] + \
                    [co[0] for co, fa in zip(coarse.items(), fame.items()) if co[1]=='person' and fa[1]=='famous'] + \
                    [co[0] for co, fa in zip(coarse.items(), fame.items()) if co[1]=='place' and fa[1]=='familiar'] + \
                    [co[0] for co, fa in zip(coarse.items(), fame.items()) if co[1]=='place' and fa[1]=='famous']
            vectors = {k : k_i for k_i, k in enumerate(ordered_names)}
        vectors = minus_one_one_norm(vectors.items())

    elif args.input_target_model == 'word_length':
        vectors = {w : len(w) for w in names}
        vectors = minus_one_one_norm(vectors.items())

    elif args.input_target_model == 'syllables':
        vectors = {w : sum([syllables.estimate(part) for part in w.split()]) for w in names}
        vectors = minus_one_one_norm(vectors.items())

    elif args.input_target_model == 'orthography':
        vectors = {k_one : numpy.average([levenshtein(k_one,k_two) for k_two in names if k_two!=k_one]) for k_one in names}
        vectors = minus_one_one_norm(vectors.items())

    elif args.input_target_model in ['imageability', 'familiarity']:
        fams = list()
        if args.experiment_id == 'two':
            raise RuntimeError('For this experiment familiarity is called sentence_lengths')
        else:
            filename = os.path.join('lab','stimuli',
                                    '{}_ratings_experiment.csv'.format(args.input_target_model))
            with open(filename) as i:
                lines = [l.strip().split('\t')[1:] for l in i.readlines()]
            assert len(names) <= len(lines[0])
            for k in names:
                try:
                    assert k in lines[0]
                    rel_index = lines[0].index(k)
                    try:
                        fam = numpy.average([int(l[rel_index]) for l in lines[1:]])
                    except IndexError:
                        fam = float(lines[-1][-1])
                except AssertionError:
                    #fam = 3.5
                    fam = 4.5
                fams.append(fam)

        vectors = {k_one : l_one for k_one, l_one in zip(names, fams)}
        vectors = minus_one_one_norm(vectors)

    elif args.input_target_model in ['sentence_lengths']: 
        ### reading file
        with open(os.path.join(
                               'all_models', 
                               'exp_{}_{}_{}_ratings.tsv'.format(args.experiment_id, args.input_target_model, args.language),
                              )) as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
        for l in lines:
            assert len(l) == 3
        vectors = {l[1] : float(l[2]) for l in lines if int(l[0]) == n}
        vectors = minus_one_one_norm(vectors.items())

    elif 'frequency' in args.input_target_model:
        with open(os.path.join(
                               'all_models', 
                               'exp_{}_{}_wikipedia_full_corpus_{}_ratings.tsv'.format(args.experiment_id, args.input_target_model, args.language),
                              )) as i:
            lines = [l.strip().split('\t') for l in i.readlines()][1:]
        vectors = {l[0] : float(l[1]) for l in lines}
        vectors = minus_one_one_norm(vectors.items())

    elif args.input_target_model in [ 
                               'ITGPT2', 
                               'gpt2-xl', 
                               'xlm-roberta-large',
                               'MBERT',
                               'BERT_large',
                               'wikipedia2vec',
                               'transe',
                               'w2v',
                             ]:
        ### reading file
        with open(os.path.join(
                               'all_models', 
                               'exp_{}_{}_{}_vectors.tsv'.format(args.experiment_id, args.input_target_model, args.language),
                              )) as i:
            lines = [l.strip().split('\t') for l in i.readlines()][1:]
        vectors = {l[0] : numpy.array(l[1:], dtype=numpy.float64) for l in lines}
        for k, v in vectors.items():
            #print(v.shape)
            assert v.shape in [(300, ), (1024, ), (1280,), (768,), (1600,), (500,), (100,)]

    elif 'gold' in args.input_target_model or 'individuals' in args.input_target_model or 'model' in args.input_target_model or 'random' in args.input_target_model or '_all' in args.input_target_model or '_one' in args.input_target_model:
        ### reading file
        file_path = os.path.join(
                               'all_models', 
                               #'exp_{}_{}_{}_average_{}_entity_articles_limit_1seventh_vectors.tsv'.format(args.experiment_id, args.input_target_model, args.language, args.average))
                               #'exp_{}_{}_{}_average_{}_entity_articles_limit_1fifth_vectors.tsv'.format(args.experiment_id, args.input_target_model, args.language, args.average))
                               #'exp_{}_{}_{}_average_{}_entity_articles_limit_1fourth_vectors.tsv'.format(args.experiment_id, args.input_target_model, args.language, args.average))
                               'exp_{}_{}_{}_average_{}_entity_articles_limit_all_vectors.tsv'.format(args.experiment_id, args.input_target_model, args.language, args.average))
        #if args.input_target_model in ['BERT_large_individuals', 'ITGPT2_individuals', 'xlm-roberta-large_individuals']:
        if args.experiment_id == 'two':
            file_path = file_path.replace('articles', 'sentences')
        if args.experiment_id == 'one':
            file_path = file_path.replace('entity_articles', 'full_corpus')

        #file_path = file_path.replace('limit_20', 'exp')
        print(file_path)
        if args.average == -36:
            file_path = file_path.replace('-36', '-12')
        if 'individuals' in args.input_target_model or 'random' in args.input_target_model or '_all'  in args.input_target_model or '_one' in args.input_target_model:
            file_path = file_path.replace('_300-500ms', '')
            file_path = file_path.replace('_500-800ms', '')
        assert os.path.exists(file_path)
        with open(file_path) as i:
            lines = [l.strip().split('\t') for l in i.readlines()][1:]
        vectors = {l[1] : numpy.array(l[2:], dtype=numpy.float64) for l in lines if int(l[0]) == n}
           
        for k, v in vectors.items():
            #print(v.shape)
            #assert v.shape in [(300, ), (1024, ), (768,), (1600,), (500,), (100,)]
            assert v.shape in [(3, ), (5, ), (1, ), (300, ), (1024, ), (1280,), (768,), (1600,), (500,), (100,)]
        if v.shape in [(1, )]:
            vectors = {k : v[0] for k, v in vectors.items()}
            vectors = minus_one_one_norm(vectors.items())
        ### print correlation with length

        #lengths = [abs(len(n)-len(n_two)) for n in names for n_two in names if n!=n_two]
        #distances = [1 - scipy.stats.pearsonr(vectors[n], vectors[n_two])[0] for n in names for n_two in names if n!=n_two]
        #corr = scipy.stats.pearsonr(lengths, distances)
        #print('correlation model - length: {}'.format(corr))
    if args.experiment_id == 'one' and args.semantic_category_two != 'individual':
        pass
    else:
        for idx, cat in [(2, 'famous/familiar'), (1, 'person/place')]:
            famous_familiar = {v[0] : 0 if v[idx] in ['famous', 'person'] else 1 for v in experiment.trigger_to_info.values()}
            keyz = sorted(set(famous_familiar.keys()) & set(vectors.keys()))
            if args.experiment_id == 'one':
                assert len(keyz) == 40
            else:
                assert len(keyz) == 32
            #print(len(keyz))
            fam_fam_sims = [1 if famous_familiar[one]==famous_familiar[two] else 0 for one in keyz for two in keyz if one!=two] 
            if type(vectors[keyz[0]]) in [int, float, numpy.float64] or vectors[keyz[0]].shape == (1, ):
                model_sims = [1-abs(float(vectors[one])-float(vectors[two])) for one in keyz for two in keyz if one!=two] 
            else:
                model_sims = [scipy.stats.pearsonr(vectors[one], vectors[two])[0] for one in keyz for two in keyz if one!=two] 
            corr = scipy.stats.pearsonr(fam_fam_sims, model_sims)
            print('correlation with {} distinction: {}'.format(cat, corr))

    return vectors

def zero_one_norm(vectors):
    values = [v for k, v in vectors.items()]
    values = [(v - min(values))/(max(values)-min(values)) for v in values]
    vectors = {k[0] : val for k, val in zip(vectors.items(), values)}
    return vectors

def minus_one_one_norm(vectors):
    labels = [k[1] for k in vectors]
    names = [k[0] for k in vectors]
    norm_labels = [int(2*((x-min(labels))/(max(labels)-min(labels)))-1) for x in labels]
    assert min(norm_labels) == -1
    assert max(norm_labels) == 1
    vectors = {n : l for n, l in zip(names, labels)}
    return vectors
