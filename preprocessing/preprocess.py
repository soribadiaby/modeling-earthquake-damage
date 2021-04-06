import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from argparse import ArgumentParser
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from . import Dataset, DataModule, Autoencoder


class Preprocess:
    
    GEOLEVELS_RANGE = [31, 1414, 11595]

    def __init__(self) -> None:
        self.args      = self.parse_args()
        self.dataset   = Dataset(self.args.rootdir)
        
    @staticmethod
    def parse_args():
        parser = ArgumentParser()
        parser = Trainer.add_argparse_args(parser)
        parser.add_argument("rootdir", help="path to the folder containing the csv files")
        parser.add_argument("-o", "--outputdir",
                            help="path to the folder in which to store preprocessing \
                            results, created if needed.",
                            default="./preprocessing_outputs/")
        parser.add_argument("-b", "--batch_size",
                            help="batch size for autoencoder training",
                            type=int, default=64)
        parser.add_argument("-n", "--num_workers",
                            help="number of threads for data loading",
                            type=int, default=8)
        parser.add_argument("--embedding_dim",
                            help="autoencoder embedding dimension",
                            type=int, default=512)
        parser.add_argument("--hidden_dim",
                            help="autoencoder hidden dimension (ie number of features)",
                            type=int, default=16)
        parser.add_argument('-s','--steps', nargs='+', help='Preprocessing steps',
                            type=int, choices=[1, 2], default=[1, 2])
        return parser.parse_args()

    def create_output_dir_if_needed(self) -> None:
        if not os.path.exists(self.args.outputdir):
            os.makedirs(self.args.outputdir)

    def add_conditional_probabilites(self, feature: str, length: int) -> None:
        features    = pd.merge(self.dataset.X, self.dataset.y, on='building_id')
        num_samples = {1: {}, 2: {}, 3: {}}
        damages     = {1: {}, 2: {}, 3: {}}
        for i, j in tqdm((self.dataset.X[feature].value_counts()).iteritems(), total=length):
            for k in range(3):
                num_samples[k+1] = len(features[features['damage_grade']==k+1][features[feature]==i])
                damages[k+1][i]  = num_samples[k+1] / j
        for dataframe in [self.dataset.X, self.dataset.test]:
            probas = {1: [], 2: [], 3: []}
            for i in dataframe[feature]:
                for j in range(3):
                    probas[j+1].append(damages[j+1].get(i))
            for i in range(3):
                dataframe[f'prob{i+1}_{feature}'] = probas[i+1]

    def add_all_conditional_probabilites(self) -> None:
        print("Step 1: Adding Conditional Probabilites...")
        for i in range(3):
            print(f"    * Handling geolevel {i+1}...")
            self.add_conditional_probabilites(f'geo_level_{i+1}_id', self.GEOLEVELS_RANGE[i])

    def init_trainer(self):
        """ Init a Lightning Trainer using from_argparse_args
        Thus every CLI command (--gpus, distributed_backend, ...) become available.
        """
        lr_logger      = LearningRateMonitor()
        early_stopping = EarlyStopping(monitor   = 'val_loss',
                                    mode      = 'min',
                                    min_delta = 0.001,
                                    patience  = 10,
                                    verbose   = True)
        trainer = Trainer.from_argparse_args(self.args, default_root_dir=self.args.outputdir,
                                             callbacks = [lr_logger, early_stopping])
        return trainer

    def run_training(self):
        """ Instanciate a datamodule, a model and a trainer and run trainer.fit(model, data) """
        print("Step 2: Training Autoencoder...")
        data    = DataModule(self.args.rootdir, self.args.batch_size, self.args.num_workers)
        model   = Autoencoder(self.args.embedding_dim, self.args.hidden_dim)
        trainer = self.init_trainer()
        trainer.fit(model, data)
        return model

    def _add_features(self, dataframe, dataset, model, output_name):
        features = []
        with torch.no_grad():
            for data in tqdm(dataset):
                data = torch.as_tensor(data, dtype=torch.long)
                if self.args.gpus > 0:
                    data = data.cuda()
                _ = model(data)
                features.append(model.features.cpu().numpy())
        features = np.array(features)
        new_df = pd.get_dummies(dataframe.copy())
        new_df = new_df.drop(['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id'], axis=1)
        new_data = np.hstack((np.array(new_df), features))
        np.savez(os.path.join(self.args.outputdir, output_name), data=new_data)

    def add_autoencoder_features(self, autoencoder):
        print("Step 3: Adding Autoencoder features and saving preprocessed data...")
        dm = DataModule(self.args.rootdir)
        dm.setup("test")
        if self.args.gpus > 0:
            autoencoder = autoencoder.cuda()
        autoencoder.eval()
        train_length = dm.test_set.X.shape[0]
        self._add_features(self.dataset.X, dm.test_set.geolevel3[:train_length],
                           autoencoder, "train_data")
        self._add_features(self.dataset.test, dm.test_set.geolevel3[train_length:],
                           autoencoder, "test_data")

    def save_preprocessed_csv(self):
        self.dataset.X.to_csv(   os.path.join(self.args.outputdir, 'train_values.csv'))
        self.dataset.y.to_csv(   os.path.join(self.args.outputdir, 'train_labels.csv'))
        self.dataset.test.to_csv(os.path.join(self.args.outputdir, 'test_values.csv'))

    def run(self) -> None:
        print("Preprocessing...")
        start = time()
        self.create_output_dir_if_needed()
        if 1 in self.args.steps:
            self.add_all_conditional_probabilites()
            self.save_preprocessed_csv()
        if 2 in self.args.steps:
            autoencoder = self.run_training()
            self.add_autoencoder_features(autoencoder)
        stop = time()
        preprocessing_time = stop - start
        print("Preprocessing finished ! ")
        print(f"Time taken: {preprocessing_time // 60} minutes, {preprocessing_time % 60} seconds.")
