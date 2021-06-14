from preprocessing.preprocessing_classification import clean_dataset
from functools import cached_property
from tqdm import tqdm
import numpy as np
import typing
import pickle
import os


class Glove:
    def __init__(self, glove_txt_prefix: str, dim=100):
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
        with open(path + 'pkl', 'wb') as f:
            pickle.dump(embeddings, f)
        return embeddings

    def tokens_embedding(self, tokens: typing.List[str]) -> typing.Optional[np.ndarray]:
        return np.array(list(map(self.embeddings.get, tokens)))
