# Imports for plotting
import os
from copy import deepcopy

import lightning as l
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from matplotlib.colors import to_rgb
from torch.utils.data import DataLoader

import audio_deep_learning_config as cfg

plt.set_cmap('cividis')

matplotlib.rcParams['lines.linewidth'] = 2.0

sns.reset_orig()

"""this class is for the datamodule of the model"""
class DataModule(l.LightningDataModule):

    # Initializer
    def __init__(self):
        super().__init__()
        self.data_dir = cfg.default_data_dir
        self.batch_size_train = cfg.batch_size_train
        self.batch_size_test = cfg.batch_size_test
        self.num_workers = cfg.num_workers
        self.dataset = cfg.dataset

    # prepare data
    # must be implemented in the subclass
    def prepare_data(self) -> None:
        return super().prepare_data()

    # setup data
    # must be implemented in the subclass
    def setup(self, stage: str):
        print(stage)
        if stage == "fit":
            self.dataset_train = self.dataset(
                data_dir = os.path.join(self.data_dir, "train"),
                num_data = 5000,
            )
            self.dataset_val = self.dataset(
                data_dir = os.path.join(self.data_dir, "dev"),
                num_data = 998,
            )
        elif stage == "test":
            self.dataset_test = self.dataset(
                data_dir = os.path.join(self.data_dir, "test"),
                num_data = 5000
            )

    # train dataloaders settings
    # must be implemented in the subclass
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.dataset_train,
            batch_size = self.batch_size_train,
            shuffle = True,
            num_workers = self.num_workers,
            pin_memory = False
        )

    # evaluation dataloaders settings
    # must be implemented in the subclass
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.dataset_val,
            batch_size = self.batch_size_test,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = False
        )

    # test dataloaders settings
    # must be implemented in the subclass
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.dataset_test,
            batch_size = self.batch_size_test,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = False
        )

# main function to be runned
def cli_main():
    cli = LightningCLI(
        cfg.model,
        DataModule,
        seed_everything_default=1744,
        save_config_kwargs={'overwrite': True},
        # parser_kwargs={"default_config_files": ["config/default.yaml"],
        #    "parser_mode": "omegaconf"
        #    },
    )
    # model = TransformerPredictor()
    # data = DataModule()

    # trainer = l.Trainer(
    #     accelerator="auto",
    #     fast_dev_run=10,
    # )
    # trainer.fit(model, data)

if __name__ == '__main__':
    cli_main()
