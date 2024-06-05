# Imports for plotting

from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from lightning.pytorch.cli import LightningCLI

from matplotlib.colors import to_rgb

import audio_deep_learning_config as cfg
from module import TrustedRCNN

plt.set_cmap('cividis')

matplotlib.rcParams['lines.linewidth'] = 2.0

sns.reset_orig()

# main function to be runned
def cli_main():
    cli = LightningCLI(
        cfg.model,
        cfg.data_m,
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
