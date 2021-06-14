import pandas as pd
import re
import numpy as np
import pickle
import os
from feature_extraction import tokenize_nopunct


def __clean_lyrics(lyrics):
    lyrics = re.sub(r'\[[^\[]*?\]', '', lyrics)  # remove things like [Chorus] [First verse]
    lyrics = re.sub(r'\r?\n|\r', '\n', lyrics)  # transform white lines in \n
    lyrics = re.sub(r'\n+', '\n', lyrics)  # remove duplicated \n
    return lyrics


def _fetch_lyrics(file_path):
    df = pd.read_csv(file_path)
    df = df[df['lyrics'].notna()]  # remove NaN
    df = df[df['lyrics'].str.split().str.len() <= 2000]  # remove books(?)
    df = df[df['lyrics'].str.split().str.len() > 5]  # remove almost-empty strings
    df = df[df.groupby('artist').artist.transform('count') > 10]  # leaves only artists with n+ songs
    df = df[df['artist'] != 'Glee Cast']  # bravi ma non orginali
    dataset = {}
    lyrics = df['lyrics']
    dataset['lyrics'] = np.array([__clean_lyrics(lyric) for lyric in lyrics])
    dataset['artists'] = np.array(df['artist'])
    dataset['songs'] = np.array(df['song'])
    return dataset


def fetch_dataset(pickle_path, lyrics_path, force=False):
    if os.path.exists(pickle_path) and force is False:
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        try:
            os.remove(pickle_path)
        except OSError:
            pass
        dataset = _fetch_lyrics(lyrics_path)
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset, f)
    return dataset
