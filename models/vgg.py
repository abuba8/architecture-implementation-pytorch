import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=(4,4), stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(4,4), stride=(1,1))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(4,4), stride=(1,1))
        self.tanh = nn.Tanh()
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.tanh(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)

        return x

# x = torch.randn(64,1, 32, 32)
# model = LeNet()
# print(model(x).shape)

        