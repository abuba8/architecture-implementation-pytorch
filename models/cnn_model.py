import torch.nn as nn
import torch.nn.functional as _f

class CNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc = nn.Linear(16*7*7, n_classes)

    def forward(self, x):
        x = _f.relu(self.conv1(x))
        x = self.pool(x)
        x = _f.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x