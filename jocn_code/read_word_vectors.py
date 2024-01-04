import numpy

from tqdm import tqdm

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

    ### categories / familiarity
    if args.input_target_model in ['coarse_category', 'famous_familiar']:
        if args.input_target_model == 'coarse_category':
            categories = {v[0] : v[1] for v in experiment.trigger_to_info.values()}
            sorted_categories = sorted(set(categories.values()))
        elif args.input_target_model == 'famous_familiar':
            categories = {v[0] : v[2] for v in experiment.trigger_to_info.values()}
            sorted_categories = sorted(set(categories.values()))
        vectors = {w : sorted_categories.index(categories[w]) for w in names}
        vectors = minus_one_one_norm(vectors.items())
    # orthographic values
    elif args.input_target_model == 'word_length':
        vectors = {w : len(w) for w in names}
        vectors = minus_one_one_norm(vectors.items())

    elif args.input_target_model == 'orthography':
        vectors = {k_one : numpy.average([levenshtein(k_one,k_two) for k_two in names if k_two!=k_one]) for k_one in names}
        vectors = minus_one_one_norm(vectors.items())

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
