import numpy.random
import pandas as pd
import re
import numpy as np
import pickle
import os
import typing


genres_acc = ['pop', 'progressive rock', 'rock', 'metal', 'country', 'rnb', 'funk', 'hip-hop', 'alternative',
              'rap', 'disco', 'folk', 'jazz', 'blues', 'indie', 'christmas', 'soul', 'reggae', 'gospel', 'latin']


def _get_genre(sub_genres):
    sub_genres = sub_genres.split(':')
    for sub_genre in sub_genres:
        sub_genre = sub_genre.lower()
        if sub_genre == 'hard rock':
            return genres_acc.index('metal')
        if sub_genre in ['electronic', 'dance', 'dancehall']:
            return genres_acc.index('disco')
        if 'christian' in sub_genre:
            return genres_acc.index('gospel')
        if 'hip hop' in sub_genre or 'hiphop' in sub_genre:
            return genres_acc.index('hip-hop')
        if 'spanish' in sub_genre:
            return genres_acc.index('latin')
        if sub_genre == 'r&b' or 'doo wop' in sub_genre:
            return genres_acc.index('rnb')
        for genre in genres_acc:
            if genre in sub_genre:
                return genres_acc.index(genre)
    return 'wtf'


def _clean_lyrics(lyrics):
    lyrics = re.sub(r'\[[^\[]*?\]', '', lyrics)  # remove things like [Chorus] [First verse]
    lyrics = re.sub(r'\{[^\{]*?\}', '', lyrics)  # remove things like [Chorus] [First verse]
    lyrics = re.sub(r'\r?\n|\r', '\n', lyrics)  # transform white lines in \n
    lyrics = re.sub(r'\n+', '\n', lyrics)  # remove duplicated \n
    return lyrics


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.lyrics.notna()]  # remove NaN
    df = df[df.genre.notna()]
    df = df[df.lyrics.str.split().str.len() <= 2000]  # remove books(?)
    df = df[df.lyrics.str.split().str.len() > 5]  # remove almost-empty strings
    df = df[df.groupby('artist').artist.transform('count') > 10]  # leaves only artists with n+ songs
    df = df[df['artist'] != 'Glee Cast']  # bravi ma non orginali
    df['lyrics'] = df['lyrics'].apply(_clean_lyrics)
    df['genre'] = df['genre'].apply(_get_genre)
    df = df[df.genre != 'wtf']
    return df


def df_as_dict(df: pd.DataFrame) -> typing.Dict[str, np.ndarray]:
    dataset = {}
    dataset['lyrics'] = np.array(df.lyrics)
    dataset['artists'] = np.array(df.artist)
    dataset['songs'] = np.array(df.song)
    dataset['genres'] = np.array(df.genre)
    return dataset


def select_random_authors(df, n_authors):
    authors = np.unique(df['artist'])
    np.random.seed(42)
    selected_authors = np.random.choice(authors, n_authors, replace=False)
    df = df.loc[df['artist'].isin(selected_authors)]
    return df


def fetch_dataset(pickle_path, lyrics_path, force=False, as_dict=False, random_authors=0):
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
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset, f)
    if random_authors != 0:
        dataset = select_random_authors(dataset, random_authors)
    if as_dict:
        dataset = df_as_dict(dataset)
    return dataset
