import sys
sys.path.insert(1, '/mnt/KInsektDaten/teo/pytorch-forecasting')
import os

import pickle
from lightning.pytorch import seed_everything
from lightning.pytorch.strategies import DDPStrategy
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use("Agg") # to prevent the allocate bitmap warning

from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, MAPE, SMAPE, RMSE, QuantileLoss

# Setup
#BASE_PATH = 'D:/KIMoDIs/global-groundwater-models-main'
DATA_PATH = '../../../data/Grundwasser/kimodis_preprocessed/data/'
#MODEL_PATH = os.path.join(BASE_PATH, 'models')
#RESULT_PATH = os.path.join(BASE_PATH, 'results')
SHARE_PATH = '../../../data/Grundwasser/kimodis_preprocessed/'

# Specify model type
#MODEL_TYPE = 'dyn' # full # full_interpol
BATCH_SIZE = 32 #4096
VERSION = '10_Epochs'

# Model types:
# - full: all input features  
# - full_interpol: all input features except for the gw level, doesn't work with N-HiTS. 
# Respective Error:
# [self.hparams.x_reals.index(name) for name in to_list(target)],
# ValueError: 'gwl' is not in list
# - dyn: only dynamic input features
MODEL_TYPE_ls = ['full']#, 'full_interpol','dyn']
seeds = [94]#, 673, 1899, 2149, 9898]

for MODEL_TYPE in MODEL_TYPE_ls: 
    
    # Load TFT TimeSeriesDataSets and create dataloaders
    train_ds = TimeSeriesDataSet.load(os.path.join(SHARE_PATH, f'train_ds_{MODEL_TYPE}_tft.pt'))
    print(len(train_ds.decoded_index['proj_id'].unique()), 'sites are in the training data.')
    train_dataloader = train_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=18)

    val_ds = TimeSeriesDataSet.load(os.path.join(SHARE_PATH, f'val_ds_{MODEL_TYPE}_tft.pt'))
    print(len(val_ds.decoded_index['proj_id'].unique()), 'sites are in the validation data.')
    val_dataloader = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=18)
    
    for i in seeds:
        seed_everything(i, workers = True)

        # Create study for HPO
        study_file = f'../models/tft/optuna_checkpoints/tft_{MODEL_TYPE}_teo/tft_{MODEL_TYPE}_teo.pkl'
        study = optimize_hyperparameters(
            train_dataloader,
            val_dataloader,
            model_path=f'../models/tft/optuna_checkpoints/tft_{MODEL_TYPE}_teo',
            log_dir=f'../models/tft/optuna_checkpoints/tft_{MODEL_TYPE}_teo/lightning_logs',
            # study=study_file, # to resume HPO after crashing
            loss=QuantileLoss(),
            n_trials=5,
            max_epochs=10,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(4, 16),
            hidden_continuous_size_range=(4, 16),
            attention_head_size_range=(1, 4),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=True,
            verbose=2
        )

        # Save study results - also we can resume tuning at a later point in time
        with open(study_file, "wb") as fout:
            pickle.dump(study, fout)

        # Show best hyperparameters
        print(study.best_trial.params)
