# Some useful configuration parameters for audio deep learning project

# The default path in the container used for data storage
default_data_dir = '/workspaces/data'

# The batch size for training and testing
batch_size_train = 2
batch_size_test = 1

# The number of workers for data loading
num_workers = 8

# The dataset class to use. This class must inherrite from torch.utils.data.Dataset
# This class should be implemented in the dataloader/dataset.py file, or imported otherwise.
# dataset = dataset.myDataset()
from dataloader import dataset as ds
dataset = ds.TSSLDataSet()

# The model class to use. This class must inherrite from LightningModule
from model import TrustedRCNN
model = TrustedRCNN
