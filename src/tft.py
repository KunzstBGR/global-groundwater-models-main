import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

# import pytorch_lightning as pl
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.strategies import DDPStrategy
# from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, StochasticWeightAveraging, LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use("Agg") # to prevent the allocate bitmap warning

from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting import TimeSeriesDataSetW
from pytorch_forecasting.metrics import MAE, MAPE, SMAPE, RMSE, QuantileLoss

# Setup
BASE_PATH = 'D:/KIMoDIs/global-groundwater-models-main'
DATA_PATH = os.path.join(BASE_PATH, 'data')
MODEL_PATH = os.path.join(BASE_PATH, 'models')
RESULT_PATH = os.path.join(BASE_PATH, 'results')
SHARE_PATH = 'J:/Berlin/B22-FISHy/NUTZER/Kunz.S/kimodis_preprocessed'

# Specify model type
BATCH_SIZE = 4096
VERSION = '10_Epochs'

# Model types:
# - full: all input features  
# - full_interpol: all input features except for the gw level 
# - dyn: only dynamic input features
MODEL_TYPE_ls = ['full', 'full_interpol' ,'dyn'] 
seeds = [40, 94, 673, 1899, 2100, 2149, 2230, 6013, 9595, 9898]

for MODEL_TYPE in MODEL_TYPE_ls: 

    # Load TFT TimeSeriesDataSets and create dataloaders
    train_ds = TimeSeriesDataSet.load(os.path.join(SHARE_PATH, f'train_ds_{MODEL_TYPE}_tft.pt'))
    print(len(train_ds.decoded_index['proj_id'].unique()), 'sites are in the training data.')
    train_dataloader = train_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)

    val_ds = TimeSeriesDataSet.load(os.path.join(SHARE_PATH, f'val_ds_{MODEL_TYPE}_tft.pt'))
    print(len(val_ds.decoded_index['proj_id'].unique()), 'sites are in the validation data.')
    val_dataloader = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    
    for i in seeds:
        seed_everything(i, workers = True)
    
        # Data distributed parallel
        # Change the backend, see: 
        #  https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html
        ddp = DDPStrategy(process_group_backend='gloo')

        # Model
        # https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html
        # log interval: log predictions every x batches
        # log val interval: defaults to log interval
        # default learning rate: 0.001
        # TODO: Find optimal learning rate
        # Logging metrics: SMAPE, MAE, RMSE, MAPE
        # Metrics: 
        # https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.metrics.point.RMSE.html#
        lr = 0.003981 # only parameter that has been optimized
        model = TemporalFusionTransformer.from_dataset(train_ds, 
                                                      loss=QuantileLoss(), 
                                                      dropout=0.2, 
                                                      learning_rate=lr) # QuantileLoss()

        # Logger
        tb_logger = TensorBoardLogger(save_dir=f'D:/KIMoDIs/global-groundwater-models-main/models/tft/tft_{MODEL_TYPE}', 
                                      name=f'tft_{MODEL_TYPE}_{BATCH_SIZE}_{VERSION}')

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

        # Schedule to reduce the learning rate throughout training
        lr_sched = StochasticWeightAveraging(swa_lrs=lr, swa_epoch_start=2, device=torch.device('cuda:0'))
        lr_logger = LearningRateMonitor(logging_interval='step')  # log the learning rate ('step' or 'epoch')

        # Training
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html
        trainer = pl.Trainer(
            max_epochs=10,
            accelerator='gpu', 
            gradient_clip_val=0.2, #Prevent overfitting
            devices=4, #GPU devices
            enable_model_summary=True,
            strategy=ddp,
            fast_dev_run=True,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=tb_logger,
            log_every_n_steps=10,  # Default = 50 (steps)
            val_check_interval=0.2 # Check after 50% of each epoch
        )
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

# Access tensorboard:
# tensorboard --logdir=version_ ....

# Include proxy to install packages on kidev02
# !pip --proxy http://wwwproxy.nlfb.bgr.de:8080 install tensorflow==2.9