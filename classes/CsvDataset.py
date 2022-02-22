
import torch as T
from sklearn import preprocessing
from pandas import read_csv
import numpy as np


class CsvDataset(T.utils.data.Dataset):
    def __init__(self, n=None, df=None, features=None, target=None,
                 scale=False, scale_axis=0, normalize=True, normalize_axis=0):

        assert df is not None, "empty dataframe"

        df = df.to_numpy()
        if n is not None:
            self.y = df[:, n]
            self.X = df[:, 0:n]
        else:
            self.y = df[:, target]
            self.X = df[:, features]

        if scale is True:
            self.X = preprocessing.scale(self.X, axis=scale_axis)
        if normalize is True:
            self.X = preprocessing.normalize(self.X, axis=normalize_axis)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

