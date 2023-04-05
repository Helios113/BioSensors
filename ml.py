import torch.nn as nn
import torch


class PINN(nn.Module):
    def __init__(self, aa_num = 2) -> None:
        super(PINN, self).__init__()
        
        self.aa_num = aa_num

        self.model = nn.Sequential(
            nn.Linear(5001, 5000*250),
            nn.Tanh(),
            nn.Linear(5000*250, 5000*500),
            nn.ReLU(),
            nn.Linear(5000*500, 5000*1000)
        )


    def forward(self, x):
        sum = 0
        for i in range(self.aa_num):
            sum +=torch.div(torch.linalg.cross(self.model(x[i]), x[i]), self.aa_num)
        return sum


