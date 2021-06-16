from preprocessing import fetch_dataset
from classification import kcv_classification, AV_classification
import numpy as np
import random


dataset = fetch_dataset(pickle_path='data/dataset.pickle', lyrics_path='data/results_genre.csv', force=True, as_dict=True, random_authors=50)

print('#samples:', len(dataset['lyrics']))
print('#artists:', len(np.unique(dataset['artists'])))
print('#genres:', len(np.unique(dataset['genres'])))

AV_classification(dataset, domain='artists', feat='base')

