import torch.nn as nn
import torch
from time import sleep


class FakeRewardModel(nn.Module):
    def __init__(self, num_features) -> None:
        super().__init__()
        self.fc1 = nn.Linear(num_features, 10)
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x):
        sleep(2)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class FakeLLM(nn.Module):
    def __init__(self, sent_len, device):
        super().__init__()
        self.sent_len = sent_len
        self.device = device
        self.lin = nn.Linear(sent_len, 1)
    
    def forward(self, x):
        sleep(0.2)
        return self.lin(x)

    def generate(self, batch_size=1):
        sleep(1.5)
        return torch.rand((batch_size, self.sent_len), device=self.device)

