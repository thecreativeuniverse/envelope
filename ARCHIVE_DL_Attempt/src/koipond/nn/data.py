import torch
import shutil
import pandas
import os

# ref: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
class KoiPond(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_dir):
        'Initialization'
        self.data_dir = data_dir
        labels_df = pandas.read_csv(f"{data_dir}/init/labels.csv")
        self.list_IDs = labels_df['curve_id']
        self.labels = labels_df['label']

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load(f"{self.data_dir}/{ID}.pt")
        X = X.view(1, len(X))
        y = torch.tensor(self.labels[index])
        return (X.float(), y.float())
