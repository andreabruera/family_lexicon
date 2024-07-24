import numpy
import os
import random
import scipy

from scipy import stats
from tqdm import tqdm

for case in ['familiarity', 'imageability', 'length']:
    ratings = {'person' : list(), 'place' : list()}
    if case == 'length':
        f = os.path.join('ratings', 'familiarity_original_ratings_exp_two.tsv')
    else:
        f = os.path.join('ratings', '{}_original_ratings_exp_two.tsv'.format(case))
    with open(f) as i:
        for l_i, l in enumerate(i):
            if l_i == 0 or l_i>16:
                continue
            line = l.strip().split('\t')
            name = line[0]
            cat = 'person' if int(line[1])<109 else 'place'
            if case == 'length':
                ratings[cat].append(len(name))
                continue
            vals = numpy.array(line[4:], dtype=numpy.float32)
            #vals = [float(line[2])]
            ratings[cat].append(numpy.average(vals))
    assert len(ratings['person']) == 8
    assert len(ratings['person']) == len(ratings['place'])
    ### permutation test
    real_diff = abs(numpy.average(ratings['person'])-numpy.average(ratings['place']))
    fake_distr = list()
    for _ in tqdm(range(1000)):
        fake = random.sample([val for v in ratings.values() for val in v], k=16)
        fake_pers = fake[:8]
        fake_place = fake[8:]
        fake_diff = abs(numpy.average(fake_pers)-numpy.average(fake_place))
        fake_distr.append(fake_diff)
    ### p-value
    p_val = (sum([1 for _ in fake_distr if _>real_diff])+1)/(1001)
    ### t-value
    # adapted from https://matthew-brett.github.io/cfd2020/permutation/permutation_and_t_test.html
    pers_errors = [v-numpy.average(ratings['person']) for v in ratings['person']]
    place_errors = [v-numpy.average(ratings['place']) for v in ratings['place']]
    all_errors = pers_errors + place_errors
    est_error_sd = numpy.sqrt(sum([er**2 for er in all_errors]) / (len(ratings['person']) + len(ratings['place']) - 2))
    sampling_sd_estimate = est_error_sd * numpy.sqrt(1 / len(ratings['person']) + 1 / len(ratings['place']))
    print(case)
    print('overall mean: {}'.format(numpy.average([v for val in ratings.values() for v in val])))
    print('overall std: {}'.format(numpy.std([v for val in ratings.values() for v in val])))
    print('mean person values: {}'.format(numpy.average(ratings['person'])))
    print('person values std: {}'.format(numpy.std(ratings['person'])))
    print('mean place values: {}'.format(numpy.average(ratings['place'])))
    print('place values std: {}'.format(numpy.std(ratings['place'])))
    print('t-value: {}'.format(real_diff/sampling_sd_estimate))
    print('p-value: {}'.format(p_val))
