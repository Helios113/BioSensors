import torch.nn as nn
import torch

class PINN_node(nn.Module):
    def __init__(self) -> None:
        super(PINN_node, self).__init__()
        
        self.nextNN  = []
        self.prevNN  = []
        
        self.nextOut = None
        self.prevOut = None
        self.z = None
        
        self.base = nn.Sequential(
                nn.Linear(2, 20),
                nn.LeakyReLU()
            ) 
        
        self.column = nn.Sequential(
                nn.Linear(50, 40),
                nn.LeakyReLU(),
                nn.Linear(40, 20),
                nn.LeakyReLU(),
            ) 
        
        self.hidden = nn.Linear(20, 7)
                
        self.nextLayer = nn.Sequential(
                nn.Linear(20, 15),
                nn.LeakyReLU()
            ) 
              
        self.prevLayer = nn.Sequential(
                nn.Linear(20, 15),
                nn.LeakyReLU()
            ) 
        
    def setPrev(self, prev):
        self.prevNN.append(prev)

    def setNext(self, next):
        self.nextNN.append(next)
    
    def stage(self, x, i):
        x = x.unsqueeze(1).float()
        q = torch.full(x.shape,i).float()
        p = torch.cat((x,q), dim=1)
        self.z = self.base(p)
        self.nextOut = self.nextLayer(self.z)
        self.prevOut = self.prevLayer(self.z)
    
    def forward(self):
        prev = torch.zeros_like(self.nextOut)
        next = torch.zeros_like(self.nextOut)
        
        if len(self.prevNN)==1:
            prev = self.prevNN[0].nextOut
        if len(self.nextNN)==1:
            next = self.nextNN[0].prevOut
        
        res = torch.cat((prev, self.z, next), dim=1)        
        y = self.column(res)

        z = self.hidden(y)
        return z

class PINN(nn.Module):
    def __init__(self, n) -> None:
        super(PINN, self).__init__()
        
        self.models = nn.ModuleList()
        
        prev:PINN_node = None
        for i in range(n):
            a = PINN_node()
            if prev is not None:
                a.setPrev(prev)
                prev.setNext(a)
            self.models.append(a)
            prev = a
        
    def forward(self, x):
        for i,z in enumerate(x):
            self.models[i].stage(z, i)
        sum = 0
        for i,z in enumerate(x):
            sum = torch.add(sum, self.models[i]())
            
        return torch.div(sum, len(x))
        