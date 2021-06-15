from enum import unique
import pandas as pd
import re
import numpy as np
import pickle
import os
import typing
from preprocessing.feature_extraction import tokenize_nopunct
from datetime import datetime
from tqdm import tqdm

remove_chorus = re.compile(r'\[[^\[]*?\]')
white_trans = re.compile(r'\r?\n|\r')
duplicate_rem = re.compile(r'\n+')


def _clean_lyrics(lyrics: str):
    lyrics = remove_chorus.sub('', lyrics)  # remove things like [Chorus] [First verse]
    lyrics = white_trans.sub('\n', lyrics)  # transform white lines in \n
    lyrics = duplicate_rem.sub('\n', lyrics)  # remove duplicated \n
    return lyrics


def __mean_rank_join(df_to_group: pd.DataFrame, df_to_join: pd.DataFrame, num_weeks: float, mean_name: str, popul_name: str):
    group = df_to_group.groupby('song').agg({'rank': 'sum', 'weeks-on-board': 'max'})
    group[mean_name] = (group['rank'] + (101 * (num_weeks - group['weeks-on-board']))) / num_weeks
    group[popul_name] = (101 - group['rank']) * group['weeks-on-board']
    return df_to_join.join(group.drop(columns=['rank', 'weeks-on-board']), 'song')


def add_chart_info(df: pd.DataFrame, chart_path: str) -> pd.DataFrame:
    assert not (df.duplicated('song', keep=False)).any(), 'df should not have duplicated songs'
    charts = pd.read_csv(chart_path)
    charts.date = pd.to_datetime(charts.date, format='%Y-%m-%d')
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
    # Consider all dates before 1960 as 1960 and all dates after 2020 as 2020
    charts.date = charts.date.apply(lambda d: datetime(year=1960, month=d.month, day=d.day) if d.year < 1960 else d)
    charts.date = charts.date.apply(lambda d: datetime(year=2020, month=d.month, day=d.day) if d.year > 2020 else d)

    total_num_weeks = abs((df.iloc[0].date - df.iloc[-1].date).days / 7)
    df = __mean_rank_join(charts, df, total_num_weeks, 'rank_alltime', 'popul_alltime')

    # Group dates by decades
    decades = charts.groupby(pd.Grouper(key='date', freq='10YS'))
    for dt, decade in tqdm(decades, desc='Computing decades rank'):
        n_weeks = abs((decade.iloc[0].date - decade.iloc[-1].date).days / 7)
        df = __mean_rank_join(decade, df, n_weeks, f'rank_{dt.year}', f'popul_{dt.year}')

    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.lyrics.notna()]  # remove NaN
    df = df[df.lyrics.str.split().str.len() <= 2000]  # remove books(?)
    df = df[df.lyrics.str.split().str.len() > 5]  # remove almost-empty strings
    df = df[df.groupby('artist').artist.transform('count') > 5]  # leaves only artists with n+ songs
    df.lyrics = df.lyrics.apply(_clean_lyrics)
    return df


def df_as_dict(df: pd.DataFrame) -> typing.Dict[str, np.ndarray]:
    dataset = {}
    dataset['lyrics'] = np.array(df.lyrics)
    dataset['artists'] = np.array(df.artist)
    dataset['songs'] = np.array(df.song)
    return dataset


def fetch_dataset(pickle_path, lyrics_path, force=False, as_dict=False) -> typing.Union[pd.DataFrame, typing.Dict[str, np.ndarray]]:
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
            dataset = df_as_dict(dataset)
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset, f)

    return dataset
