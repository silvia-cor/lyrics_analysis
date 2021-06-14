from preprocessing import fetch_dataset
from classification import kcv_classification
import numpy as np
import random


dataset = fetch_dataset(pickle_path='data/dataset.pickle', lyrics_path='data/results_genre.csv', force=True, as_dict=True)

print('#samples:', len(dataset['lyrics']))
print('#artists:', len(np.unique(dataset['artists'])))


