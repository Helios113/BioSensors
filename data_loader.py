from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd

class BioSensorsDataset(Dataset):
    def __init__(self, path):
        self.csv = path
        self.pdFrame = pd.read_csv(path)

    def __getitem__(self, index):
        data = []
        target = []
        for i, item in enumerate(self.pdFrame.iloc[index]):
            if i == len(self.pdFrame.iloc[index])-1:
                target = item
            else:
                data.append[item]
        return data, target

    def __len__(self):
        return len(self.pdFrame.index)