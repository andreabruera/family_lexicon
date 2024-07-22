import numpy
import os
import re
import syllables

from io_utils import LoadEEG

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

def load_vectors(args, experiment, n):

    names = [v[0] for v in experiment.trigger_to_info.values()]


    ### ceiling: mapping to avg of other subjects
    if args.input_target_model == 'ceiling':
        ceiling = dict()
        for n_ceiling in range(1, 34):
            eeg_data_ceiling = LoadEEG(args, experiment, n_ceiling)
            eeg_ceiling = eeg_data_ceiling.data_dict
            for k, v in eeg_ceiling.items():
                original_shape = v.shape
                ### Adding and flattening
                if k not in ceiling.keys():
                    ceiling[k] = list()
                ceiling[k].append(v.flatten())
        comp_vectors = {k : numpy.average([vec for vec_i, vec in enumerate(v) if vec_i!=n-1], axis=0) for k, v in ceiling.items()}
        vectors = {experiment.trigger_to_info[k][0] : v.reshape(original_shape) for k, v in comp_vectors.items()}

    # orthographic values
    elif args.input_target_model == 'word_length':
        vectors = {w : len(w) for w in names}
        vectors = minus_one_one_norm(vectors.items())

    elif args.input_target_model == 'syllables':
        vectors = {w : sum([syllables.estimate(part) for part in w.split()]) for w in names}
        vectors = minus_one_one_norm(vectors.items())

    elif args.input_target_model == 'orthography':
        vectors = {k_one : numpy.average([levenshtein(k_one,k_two) for k_two in names if k_two!=k_one]) for k_one in names}
        vectors = minus_one_one_norm(vectors.items())

    # contextualized
    elif args.input_target_model in [ 
                               # contextualized
                               'xlm-roberta-large',
                               # context-specific static
                               'w2v_sentence',
                               ]:
        dataset_marker = 'entity_sentences_all_vectors'
        file_path = os.path.join(
                               'models', 
                               'exp_{}_{}_{}_{}.tsv'.format(
                                                  args.experiment_id, 
                                                  args.input_target_model, 
                                                  args.language,
                                                  dataset_marker
                                                  )
                               )
        #print(file_path)
        assert os.path.exists(file_path)
        with open(file_path) as i:
            lines = [l.strip().split('\t') for l in i.readlines()][1:]
        #vectors = {l[1] : numpy.array(l[2:], dtype=numpy.float64) for l in lines if int(l[0]) in [n, 'all']}
        vectors = {l[1] : numpy.array(l[2:], dtype=numpy.float64) for l in lines if l[1] in names and l[0] in ['{}'.format(n), 'all']}
           
        for k, v in vectors.items():
            #print(v.shape)
            assert v.shape in [
                               # w2v_sentence
                               (300, ),
                               # xlm-roberta
                               (1024, ),
                               ]

    return vectors

def zero_one_norm(vectors):
    values = [v for k, v in vectors.items()]
    values = [(v - min(values))/(max(values)-min(values)) for v in values]
    vectors = {k[0] : val for k, val in zip(vectors.items(), values)}
    return vectors

def minus_one_one_norm(vectors):
    labels = [k[1] for k in vectors]
    names = [k[0] for k in vectors]
    norm_labels = [2*((x-min(labels))/(max(labels)-min(labels)))-1 for x in labels]
    assert min(norm_labels) == -1
    assert max(norm_labels) == 1
    vectors = {n : l for n, l in zip(names, norm_labels)}
    return vectors
