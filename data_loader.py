from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd

class BioSensorsDataset(Dataset):
    def __init__(self, singles_path, combo_path):
        self.singles_path = singles_path
        self.combo_path = combo_path
        self.combo = pd.read_csv(self.combo_path, header=None, index_col=False).values
        self.singles = pd.read_csv(self.singles_path, header=None, index_col=False).values
        
        self.input = np.array(self.combo[:, 0:2], dtype='<U1').view(np.int32)
        self.output = np.array(self.combo[:, 2:], dtype='f')
        
    def __getitem__(self, index):
        return self.input[index], self.output[index]

    def __len__(self):
        return len(self.input)
    
    
    
# def __getitem__(self, index):
#     data = []
#     target = []
#     for i, item in enumerate(self.pdFrame.iloc[index]):
#         if i == len(self.pdFrame.iloc[index])-1:
#             target = item
#         else:
#             data.append[item]
#     return data, target
