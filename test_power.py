import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10240, 30000)
        self.fc2 = nn.Linear(30000, 30000)
        self.fc3 = nn.Linear(30000, 30000)
        self.fc4 = nn.Linear(30000, 1024)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

data  = torch.rand((3000, 10240), device="cuda:0")
label = torch.rand((3000, 1024), device="cuda:0")
net = Net().to("cuda:0")
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
loss = torch.nn.MSELoss()
while True:
    for i in range(0, 3000, 512):
        batch = data[i:i+512]
        lb = label[i:i+512]
        y = net(batch)
        l = loss(y, lb)
        l.backward()
        opt.step()
        opt.zero_grad()