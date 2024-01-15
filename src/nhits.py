# N-HiTS
# Tutorial: https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/nhits.html
# !pip --proxy http://wwwproxy.nlfb.bgr.de:8080 install "pytorch-forecasting[mqf2]"

# Imports ----
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

import lightning.pytorch as pl
# changed as pytorch lighning got updated -> lightning.pytorch is extensively used now
# https://stackoverflow.com/questions/76002944/model-must-be-a-lightningmodule-or-torch-dynamo-optimizedmodule-got-tem
from lightning.pytorch import seed_everything
from lightning.pytorch.strategies import DDPStrategy
# from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use("Agg") # to prevent the allocate bitmap warning

from pytorch_forecasting.models.nhits import NHiTS
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import MultiHorizonMetric
from pytorch_forecasting.metrics import MAE, MAPE, SMAPE, RMSE
from pytorch_forecasting.metrics.distributions import MQF2DistributionLoss

# Setup ----
BASE_PATH = 'D:/KIMoDIs/global-groundwater-models-main'
DATA_PATH = os.path.join(BASE_PATH, 'data')
MODEL_PATH = os.path.join(BASE_PATH, 'models')
RESULT_PATH = os.path.join(BASE_PATH, 'results')
SHARE_PATH = 'J:/Berlin/B22-FISHy/NUTZER/Kunz.S/kimodis_preprocessed'

# Specify Batch size and lead (for multivariate quantile loss)
BATCH_SIZE = 1024
VERSION = '10_Epochs'
LEAD=12

# Model types:
# - full: all input features  
# - full_interpol: all input features except for the gw level, doesn't work with N-HiTS. 
# Respective Error:
# [self.hparams.x_reals.index(name) for name in to_list(target)],
# ValueError: 'gwl' is not in list
# - dyn: only dynamic input features
MODEL_TYPE_ls = ['full','dyn'] # 'full_interpol' 
seeds = [94, 673, 1899, 2149, 9898]

for MODEL_TYPE in MODEL_TYPE_ls: 

    # Load time series dataset from share
    train_ds = TimeSeriesDataSet.load(os.path.join(SHARE_PATH,  f'train_ds_{MODEL_TYPE}_nhits.pt'))

    # Should be 5308 sites 
    print(len(train_ds.decoded_index['proj_id'].unique()), 'sites are in the training data.')
    train_dataloader = train_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)

    val_ds = TimeSeriesDataSet.load(os.path.join(SHARE_PATH, f'val_ds_{MODEL_TYPE}_nhits.pt'))
    # Should be 5308 sites
    print(len(val_ds.decoded_index['proj_id'].unique()), 'sites are in the validation data.')
    val_dataloader = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    
    for i in seeds:
        seed_everything(i, workers = True)
        # Data distributed parallel
        # On windows need to use ddp in a python script
        # ddp_notebook (for jupyter notebooks) only runs on Linux/Mac
        # Change the backend, see: 
        # https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html
        ddp = DDPStrategy(process_group_backend='gloo', 
                          find_unused_parameters=True)

        # Docs:
        # https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.nhits.NHiTS.html#pytorch_forecasting.models.nhits.NHiTS
        model=NHiTS.from_dataset(
            train_ds,
            #hidden_size=512, # size of hidden layers, default 512, min 8, max 1024; Use 32-128 if no covariates are employed
            loss=MQF2DistributionLoss(prediction_length=LEAD)
        )

        # Logger
        tb_logger = TensorBoardLogger(save_dir=f'D:/KIMoDIs/global-groundwater-models-main/models/nhits/nhits_{MODEL_TYPE}', 
                                      name=f'nhits_{MODEL_TYPE}_{BATCH_SIZE}_{VERSION}')

        # Early stopping
        # Patience is the number of checks with no improvement after which training will be stopped
        # Default: one check after every training epoch 
        # Also connected with check_val_every_n_epoch (set in trainer)
        # https://lightning.ai/docs/pytorch/2.0.4/api/lightning.pytorch.callbacks.EarlyStopping.html
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")

        # Checkpoint callback
        # Used to save the best model (or top-k best models)
        # Best model can be returned like this:
        # best_model = YourModel.load_from_checkpoint(checkpoint_callback.best_model_path)
        # trainer.test(best_model)
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

        # Training
        trainer = pl.Trainer(
            max_epochs=10,
            accelerator='gpu', 
            devices=4, 
            enable_model_summary=True,
            strategy=ddp,
            # fast_dev_run=True,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=tb_logger,
            log_every_n_steps=10,  # Default = 50 (steps)
            val_check_interval=0.2 # Check after 20% of each epoch
        )
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )