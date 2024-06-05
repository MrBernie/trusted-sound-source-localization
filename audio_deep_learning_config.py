# Some useful configuration parameters for audio deep learning project

# The default path in the container used for data storage
default_data_dir = '/workspaces/audio_deep_learning_1/tssl_data'

# The batch size for training and testing
default_batch_size_train = 2
default_batch_size_test = 1

# The number of workers for data loading
default_num_workers = 8

# The dataset class to use. This class must inherrite from torch.utils.data.Dataset
# This class should be implemented in the dataloader/dataset.py file, or imported otherwise.
# dataset = dataset.myDataset()
import dataloader.dataset as ds
dataset = ds.TSSLDataSet

# The model class to use. This class must inherrite from LightningModule
import module
model = module.TrustedRCNN

import data_module
data_m = data_module.DataModule
