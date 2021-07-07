<<<<<<< HEAD
=======
from enum import unique
>>>>>>> origin/class
import pandas as pd
import re
import numpy as np
import pickle
import os
import typing
from datetime import datetime
from tqdm import tqdm
from preprocessing.feature_extraction import tokenize_nopunct
from datetime import datetime
from tqdm import tqdm
from datetime import datetime
from tqdm import tqdm

genres_acc = ['pop', 'progressive rock', 'rock', 'metal', 'country', 'rnb', 'funk', 'hip-hop', 'alternative',
              'rap', 'disco', 'folk', 'jazz', 'blues', 'indie', 'christmas', 'soul']


def get_genre(sub_genres):
    sub_genres = sub_genres.split(':')
    for sub_genre in sub_genres:
        sub_genre = sub_genre.lower()
        if sub_genre == 'hard rock':
            return genres_acc.index('metal')
        if sub_genre in ['electronic', 'dance', 'dancehall']:
            return genres_acc.index('disco')
        if 'hip hop' in sub_genre or 'hiphop' in sub_genre:
            return genres_acc.index('hip-hop')
        if sub_genre == 'r&b' or 'doo wop' in sub_genre:
            return genres_acc.index('rnb')
        for genre in genres_acc:
            if genre in sub_genre:
                return genres_acc.index(genre)
    return None


remove_chorus = re.compile(r'\[[^\[]*?\]')
white_trans = re.compile(r'\r?\n|\r')
duplicate_rem = re.compile(r'\n+')


def _clean_lyrics(lyrics: str):
    lyrics = remove_chorus.sub('', lyrics)  # remove things like [Chorus] [First verse]
    lyrics = white_trans.sub('\n', lyrics)  # transform white lines in \n
    lyrics = duplicate_rem.sub('\n', lyrics)  # remove duplicated \n
    return lyrics


def __mean_rank_join(df_to_group: pd.DataFrame, df_to_join: pd.DataFrame, num_weeks: float, mean_name: str, popul_name: str):
    group = df_to_group.groupby('song').agg(rank=pd.NamedAgg(column='rank', aggfunc='sum'), 
                                            weeks_on_board=pd.NamedAgg(column='weeks-on-board', aggfunc='max'),
                                            popul=pd.NamedAgg(column='rank', aggfunc=lambda r: (101 - r).sum()))
    group[mean_name] = (group['rank'] + (101 * (num_weeks - group['weeks_on_board']))) / num_weeks
    group[popul_name] = group['popul']
    return df_to_join.join(group.drop(columns=['rank', 'weeks_on_board', 'popul']), 'song')


def aggregate_sixties_and_2020s(df: pd.DataFrame):
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
    # Consider all dates before 1960 as 1960 and all dates after 2019 as 2019
    df.date = df.date.apply(lambda d: datetime(year=1960, month=12, day=31) if d.year < 1960 else d)
    df.date = df.date.apply(lambda d: datetime(year=2019, month=12, day=31) if d.year > 2019 else d)
    return df


def add_chart_info(df: pd.DataFrame, chart_path: str) -> pd.DataFrame:
    assert not (df.duplicated('song', keep=False)).any(), 'df should not have duplicated songs'
    charts = pd.read_csv(chart_path)
    charts = aggregate_sixties_and_2020s(charts)
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d')

    total_num_weeks = abs((df.iloc[0].date - df.iloc[-1].date).days / 7)
    df = __mean_rank_join(charts, df, total_num_weeks, 'rank_alltime', 'popul_alltime')

    # Group dates by decades
    decades = charts.groupby(pd.Grouper(key='date', freq='10YS'))
    for dt, decade in tqdm(decades, desc='Computing decades rank'):
        n_weeks = abs((decade.iloc[0].date - decade.iloc[-1].date).days / 7)
        df = __mean_rank_join(decade, df, n_weeks, f'rank_{dt.year}', f'popul_{dt.year}')

    return df


def clean_dataset(df: pd.DataFrame, clean_genre=True, clean_lyrics=True) -> pd.DataFrame:
    df = df[df['artist'] != 'Glee Cast']  # bravi ma non orginali
    if clean_lyrics:
        df = df[df.lyrics.notna()]  # remove NaN
        df = df[df.lyrics.str.split().str.len() <= 2000]  # remove books(?)
        df = df[df.lyrics.str.split().str.len() > 5]  # remove almost-empty strings
        df['lyrics'] = df['lyrics'].apply(_clean_lyrics)
    if clean_genre:
        df = df[df.genre.notna()]
        df['genre'] = df['genre'].apply(get_genre)
        df = df[df.genre != None]
    return df[df.groupby('artist').artist.transform('count') > 5]  # leaves only artists with n+ songs


def df_as_dict(df: pd.DataFrame) -> typing.Dict[str, np.ndarray]:
    dataset = {}
    dataset['lyrics'] = np.array(df.lyrics)
    dataset['artists'] = np.array(df.artist)
    dataset['songs'] = np.array(df.song)
    dataset['genres'] = np.array(df.genre)
    dataset['rank'] = np.array(df.rank_alltime)
    dataset['popularity'] = np.array(df.popul_alltime)
    print(len(dataset['artists']))
    print(len(np.unique(dataset['artists'])))
    return dataset


def select_random_authors(df, n_authors, seed):
    authors = np.unique(df['artist'])
    np.random.seed(seed)
    selected_authors = np.random.choice(authors, n_authors, replace=False)
    df = df.loc[df['artist'].isin(selected_authors)]
    return df


def fetch_dataset(pickle_path, lyrics_path, force=False, as_dict=False, random_authors=0, seed=42) -> typing.Union[pd.DataFrame, typing.Dict[str, np.ndarray]]:
    if pickle_path is not None and os.path.exists(pickle_path) and not force:
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        df = pd.read_csv(lyrics_path)
        dataset = clean_dataset(df, clean_genre=clean_genre, clean_lyrics=clean_lyrics)
        dataset = add_chart_info(dataset, 'data/charts.csv')
        dataset = dataset.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'genres'])
        
        if pickle_path is not None:
            with open(pickle_path, 'wb') as f:
                pickle.dump(dataset, f)
    if random_authors != 0:
        dataset = select_random_authors(dataset, random_authors, seed)
    if as_dict:
        dataset = df_as_dict(dataset)
    return dataset
