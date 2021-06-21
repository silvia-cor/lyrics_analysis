from preprocessing import fetch_dataset
from classification import kcv_classification, AV_classification
import numpy as np
from preprocessing.test_preprocessing import TestPreprocessing
import statistics as stat


def round_of_kcv():
    macros = []
    micros = []
    for seed in range(10):
        dataset = fetch_dataset(pickle_path='data/dataset.pickle', lyrics_path='data/results_genre.csv', force=False,
                                as_dict=True, random_authors=5, seed=seed)
        print('\n')
        print('#samples:', len(dataset['lyrics']))
        print('#artists:', len(np.unique(dataset['artists'])))
        print('#genres:', len(np.unique(dataset['genres'])))

        ma, mi = kcv_classification(dataset, domain='genres', feat='prosody')
        macros.append(ma)
        micros.append(mi)
    print('MACRO-F1 MEAN:', round(stat.mean(macros), ndigits=3))
    print('MICRO-F1 MEAN:', round(stat.mean(micros), ndigits=3))


#round_of_kcv()

dataset = fetch_dataset(pickle_path='data/dataset.pickle', lyrics_path='data/results_genre.csv', force=False, as_dict=True, random_authors=0)
AV_classification(dataset, domain='genres', feat='prosody', save=False)

