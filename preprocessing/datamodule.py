from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
from . import Dataset


class DataModule(LightningDataModule):

    """ A Lightning Trainer uses a model and a datamodule. Here is defined a datamodule.
        It's basically a wrapper around dataloaders.
    """  
    
    def __init__(self, rootdir: str, batch_size: int=64, num_workers: int=4) -> None:
        """ Instanciate a Datamodule turn three Pytorch DataLoaders (train/val/test).
        Args:
            rootdir (str): Path to the folder containing the preprocessed npz file.
            batch_size (int, optional): Training batch size. Defaults to 64.
            num_workers (int, optional): How many subprocesses to use for data loading.
                                         Defaults to 4.
        """
        super().__init__()
        self.rootdir     = rootdir
        self.batch_size  = batch_size
        self.num_workers = num_workers

    @staticmethod
    def get_lengths(total_length):
        train_length, val_length = int(0.8*total_length), int(0.2*total_length)
        if train_length + val_length != total_length: # round error
            val_length += 1
        return train_length, val_length

    def setup(self, stage: str=None) -> None:
        """ Basically nothing more than train/val split.
        Args:
            stage (str, optional): 'fit' or 'test'.
                                   Init two splitted dataset or one full. Defaults to None.
        """
        if stage == 'fit' or stage is None:
            full_set = Dataset(self.rootdir)
            train_length, val_length = self.get_lengths(len(full_set))
            self.train_set, self.val_set = random_split(full_set, [train_length, val_length])
        if stage == 'test' or stage is None:
            self.test_set  = Dataset(self.rootdir)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, num_workers=self.num_workers, shuffle=True,
                          batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, num_workers=self.num_workers, shuffle=False,
                          batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, num_workers=self.num_workers, shuffle=False, 
                          batch_size=self.batch_size)