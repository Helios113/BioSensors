import torch
import numpy as np
from ml import PINN
from data_loader import BioSensorsDataset
from torch.utils.data import DataLoader

dataset = BioSensorsDataset("Data/data_in.csv","Data/new_data.csv")
data = DataLoader(dataset, batch_size=3, shuffle=True)

net = PINN(2)

optim = torch.optim.Adam(net.parameters(), lr=0.1)
loss = torch.nn.MSELoss()

for l in range(100):
    ls = 0
    for i,j in enumerate(data):
        optim.zero_grad()
        res = net(j[0].T)
        output = loss(res, j[1])
        print(res[0])
        print(j[1][0])
        
        output.backward()
        optim.step()
        ls+=output.item()
    print(ls/len(dataset))