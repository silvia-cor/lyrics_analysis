from preprocessing import fetch_dataset
from classification import SVM_classification
import numpy as np

dataset = fetch_dataset(pickle_path='data/dataset.pickle', lyrics_path='data/results_unzipped.csv', force=True)

print('#samples:', len(dataset['lyrics']))
print('#artists:', len(np.unique(dataset['artists'])))

SVM_classification(dataset)
