import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU, MaxPool1d, Linear, LogSoftmax


class ModelV1(nn.Module):

    def __init__(self):
        super(ModelV1, self).__init__()

        self.temp_conv = nn.Sequential(
            nn.Conv1d(in_channels=129,
                      out_channels=129,
                      kernel_size=64,
                      stride=1),
            nn.BatchNorm1d(129),
        )

        self.spatial1 = nn.Sequential(
            nn.Conv1d(in_channels=129, out_channels=20, kernel_size=10),
            nn.BatchNorm1d(20),
            nn.ELU(True),
        )
        self.spatial2 = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=1, kernel_size=10),
            nn.BatchNorm1d(1),
            nn.ELU(True),
        )
        self.avgpool1 = nn.AvgPool1d(kernel_size=10, stride=3, padding=0)
        self.avgpool2 = nn.AvgPool1d(kernel_size=10, stride=3, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.view = nn.Sequential(nn.Flatten())

        # initialize our softmax classifier
        self.fc1 = Linear(in_features=541, out_features=100)
        self.fc2 = Linear(in_features=100, out_features=1)

    def forward(self, x):
        x = self.temp_conv(x)
        x = self.spatial1(x)
        x = self.avgpool1(x)
        x = self.spatial2(x)
        x = self.avgpool2(x)
        x = self.view(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.squeeze()
        return x
