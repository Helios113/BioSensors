from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd

class BioSensorsDataset(Dataset):
    def __init__(self, singles_path, combo_path):
        self.singles_path = singles_path
        self.combo_path = combo_path
        self.combo = pd.read_csv(self.combo_path, header=None, index_col=False).values
        self.singles = pd.read_csv(self.singles_path, header=None, index_col=False)
        
        singles_dict = {}
        for i in range(self.singles.__len__()):
            singles_dict[self.singles.iloc[i][0]] = self.singles.iloc[i][1:].to_numpy(dtype='float32')
        
        
        
        
        input_letters = np.array(self.combo[:, 0:2], dtype='<U1')
        
        self.input = []
        for i in input_letters:
            line = None
            for j in i:
                if line is None:
                    line = singles_dict[j]
                else:
                    line = np.stack((line, singles_dict[j]))
                
            self.input.append(line)
        self.input = np.array(self.input)
       
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
