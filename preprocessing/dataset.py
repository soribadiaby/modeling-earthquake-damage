import os
import numpy as np
import pandas as pd
import torch
from typing import Tuple


class Dataset(torch.utils.data.Dataset):

    def __init__(self, rootdir: str) -> None:
        super(Dataset, self).__init__()
        self.rootdir   = rootdir
        self.X         = self.load_csv('train_values.csv')
        self.y         = self.load_csv('train_labels.csv')
        self.test      = self.load_csv('test_values.csv')
        self.geolevel1 = self.get_feature_array("geo_level_1_id")
        self.geolevel2 = self.get_feature_array("geo_level_2_id")
        self.geolevel3 = self.get_feature_array("geo_level_3_id")

    def load_csv(self, csv_name: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.rootdir, csv_name), index_col='building_id')

    def get_feature_array(self, feature: str, dummies: bool=False) -> np.array:
        df = pd.concat([self.X[feature], self.test[feature]])
        if dummies: 
            df = pd.get_dummies(df)
        return np.array(df)
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        input         = torch.as_tensor(self.geolevel3[index], dtype=torch.long)
        target_level1 = torch.as_tensor(self.geolevel1[index], dtype=torch.long)
        target_level2 = torch.as_tensor(self.geolevel2[index], dtype=torch.long)
        return input, target_level1, target_level2

    def __len__(self) -> int:
        return len(self.geolevel3)
