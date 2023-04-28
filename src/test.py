import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 定义网络
class Test_model(nn.Module):
    def __init__(self):
        super(Test_model, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.layer(x)

model = Test_model()

writer = SummaryWriter()
writer.add_graph(model, input_to_model=torch.randn((3,3)))
writer.add_scalar(tag="test", scalar_value=torch.tensor(1)
                    , global_step=1)
writer.close()