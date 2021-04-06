import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from typing import Tuple, Dict

torch.autograd.set_detect_anomaly(True)


class Autoencoder(pl.LightningModule):

    VOCAB_SIZES = [31, 1428, 12568]
    
    def __init__(self, embedding_dim:int, hidden_dim: int,
                 lr: float=1e-3, weight_decay: float=5e-4) -> None:
        """
        The call to the Lightning method save_hyperparameters() make every hp accessible through
        self.hparams. e.g: self.hparams.lr. It also sends them to TensorBoard.
        See the from_config class method to see them all.
        """
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(self.VOCAB_SIZES[2], embedding_dim)
        self.encoder   = nn.Linear(embedding_dim, hidden_dim)
        self.decoder1  = nn.Linear(hidden_dim, self.VOCAB_SIZES[0])
        self.decoder2  = nn.Linear(hidden_dim, self.VOCAB_SIZES[1])
        self.loss      = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.features = self.encoder(self.embedding(x))
        return self.decoder1(self.features), self.decoder2(self.features)

    def configure_optimizers(self) -> Dict:
        """ Instanciate an optimizer and a learning rate scheduler to be used during training.
        Returns:
            Dict: Dict containing the optimizer(s) and learning rate scheduler(s) to be used by
                  a Trainer object using this model. 
                  The 'monitor' key may be used by some schedulers (e.g: ReduceLROnPlateau).                        
        """
        optimizer = Adam(self.parameters(),
                         lr           = self.hparams.lr,
                         weight_decay = self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode     = 'min',
                                      factor   = 0.25,
                                      patience = 5,
                                      verbose  = False)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def shared_step(self, batch):
        inputs, targets1, targets2 = batch    
        outputs1, outputs2 = self(inputs)
        loss1 = self.loss(outputs1, targets1)
        loss2 = self.loss(outputs2, targets2)
        self.log('Train/Loss1', loss1)
        self.log('Train/Loss2', loss2)
        loss = 0.5 * (loss1 + loss2)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """ Perform the classic training step (infere + compute loss) on a batch.
        Note that the backward pass is handled under the hood by Pytorch Lightning.
        Args:
            batch (torch.Tensor): Tuple of two tensor. 
                                  One given to the network to be segmented. The other being the
                                  ground truth segmentation mask.
                                  Shapes: (batch_size, channels, depth, width, height)
            batch_idx (int): Dataset index of the batch. In range (dataset length)/(batch size).
        Returns:
            Dict: Scalars computed in this function. Note that this dict is accesible from 
                  'hooks' methods from Lightning, e.g on_epoch_start, on_epoch_end, etc...
        """
        loss = self.shared_step(batch)
        return {'loss': loss}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """ Perform the classic training step (infere + compute loss) on a batch.
        Args:
            batch (torch.Tensor): Tuple of two tensor. 
                                  One given to the network to be segmented. The other being the
                                  ground truth segmentation mask.
                                  Shapes: (batch_size, channels, depth, width, height)
            batch_idx (int): Dataset index of the batch. In range (dataset length)/(batch size).
        Returns:
            Dict: Scalars computed in this function. Note that this dict is accesible from
                  'hooks' methods from Lightning, e.g on_epoch_start, on_epoch_end, etc...
        """
        loss = self.shared_step(batch)
        return {'val_loss': loss}
