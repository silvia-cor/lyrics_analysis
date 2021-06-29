from preprocessing.preprocessing_classification import clean_dataset
from preprocessing.feature_extraction import tokenize_nopunct
from functools import cached_property
from tqdm import tqdm
import numpy as np
import pandas as pd
import typing
import pickle
import os


class Glove:
    unknown_tok = '[unk]'
    def __init__(self, glove_txt_prefix='glove.twitter.27B.', dim=100):
        self.glove_txt_prefix = glove_txt_prefix
        self.dim = dim

    @cached_property
    def embeddings(self) -> typing.Dict[str, np.ndarray]:
        path = f'{self.glove_txt_prefix}{self.dim}d.'
        if os.path.exists(path + 'pkl'):
            with open(path + 'pkl', 'rb') as f:
                return pickle.load(f)
        embeddings = {}
        with open(path + 'txt', 'r') as f:
            for line in tqdm(f):
                data = line.split(' ')
                embeddings[data[0]] = np.array([float(x) for x in data[1:]]) 
        embeddings[self.unknown_tok] = np.vstack(list(embeddings.values())).mean(axis=0)
        with open(path + 'pkl', 'wb') as f:
            pickle.dump(embeddings, f)
        return embeddings

    def tokens_embedding(self, tokens: typing.List[str], reduction='mean') -> typing.Optional[np.ndarray]:
        """
        :param tokens: List of tokens to fetch
        :param reduction: either 'mean' or 'sum', any other value will return the `token x emb_dim` matrix
        """
        emb = np.array(list(map(lambda t: self.embeddings.get(t, self.embeddings[self.unknown_tok]), tokens)))
        if reduction == 'mean':
            return emb.mean(axis=0)
        elif reduction == 'sum':
            return emb.sum(axis=0)
        return emb


def get_artist_mean_embeddings(df: pd.DataFrame, glove: Glove) -> typing.Dict[str, np.ndarray]:
    artist_mean = {}
    artists = df.drop_duplicates('artist').artist
    for artist in tqdm(artists):
        songs = np.vstack([glove.tokens_embedding(tokenize_nopunct(entry.lyrics)) for _, entry in df[df.artist == artist].iterrows()])
        artist_mean[artist] = songs.mean(axis=0)
    return artist_mean


def get_lyrics_mean_embeddings(df: pd.DataFrame, glove: Glove) -> typing.Dict[str, np.ndarray]:
    d = {}
    for _, row in tqdm(df.iterrows()):
        embeddings = glove.tokens_embedding(tokenize_nopunct(row.lyrics))
        d[row.song] = embeddings
    return d
