from preprocessing import fetch_dataset
from classification import kcv_classification
import numpy as np
import pandas as pd
from collections import Counter

#dataset = fetch_dataset(pickle_path='data/dataset.pickle', lyrics_path='data/results_unzipped.csv', force=True)

#print('#samples:', len(dataset['lyrics']))
#print('#artists:', len(np.unique(dataset['artists'])))
#print(np.unique(dataset['artists']))

df = pd.read_csv('data/results_genre.csv')

df = df[df['genre'].notna()]
genres = np.array(df['genre'])

genres_acc = ['pop', 'progressive rock', 'rock', 'metal', 'country', 'rnb', 'funk', 'hip-hop', 'alternative',
              'rap', 'disco', 'folk', 'jazz', 'blues', 'indie']

new_genres = []
for genre in genres:
    ls_genres = genre.split(':')
    for sub_genre in genre:
        sub_genre = sub_genre.lower()
        if sub_genre in genres_acc:
            new_genres.append(genres_acc.index(sub_genre))
            break
        if sub_genre in ['pop rock', 'soft rock', 'rock n roll', 'rock and roll', 'classic rock', 'rockabilly', 'alternative rock']:
            new_genres.append(genres_acc.index('rock'))
            break
        if sub_genre in ['heavy metal', 'hard rock']:
            new_genres.append(genres_acc.index('metal'))
            break
        if sub_genre in ['electronic']:
            new_genres.append(genres_acc.index('disco'))
            break
        new_genres.append('wtf')
print(len(new_genres))
