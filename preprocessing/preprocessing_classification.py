import pandas as pd
import re
import numpy as np
import pickle
import os
import typing
from preprocessing.feature_extraction import tokenize_nopunct


def _clean_lyrics(lyrics):
    lyrics = re.sub(r'\[[^\[]*?\]', '', lyrics)  # remove things like [Chorus] [First verse]
    lyrics = re.sub(r'\r?\n|\r', '\n', lyrics)  # transform white lines in \n
    lyrics = re.sub(r'\n+', '\n', lyrics)  # remove duplicated \n
    return lyrics


def clean_dataset(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    df = df[df.lyrics.notna()]  # remove NaN
    df = df[df.lyrics.str.split().str.len() <= 2000]  # remove books(?)
    df = df[df.lyrics.str.split().str.len() > 5]  # remove almost-empty strings
    df = df[df.groupby('artist').artist.transform('count') > 5]  # leaves only artists with n+ songs
    df.lyrics = df.lyrics.apply(_clean_lyrics)
    return df


def df_as_dict(df: pd.core.frame.DataFrame) -> typing.Dict[str, np.ndarray]:
    dataset = {}
    dataset['lyrics'] = np.array(df.lyrics)
    dataset['artists'] = np.array(df.artist)
    dataset['songs'] = np.array(df.song)
    return dataset


def fetch_dataset(pickle_path, lyrics_path, force=False, as_dict=False):
    if os.path.exists(pickle_path) and not force:
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        try:
            os.remove(pickle_path)
        except OSError:
            pass
        df = pd.read_csv(lyrics_path)
        dataset = clean_dataset(df)
        if as_dict:
            dataset = df_as_dict(df)
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset, f)

    return dataset
