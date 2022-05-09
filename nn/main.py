import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import sys
import configparser
sys.path.insert(1, '../')
from models.nn_model import NN as model
from utils.utils import accuracy, device


config= configparser.ConfigParser()
config.read(r'../config.ini')
input_size = int(config['nnparams']['input_size'])
n_classes = int(config['nnparams']['n_classes'])
learning_rate = float(config['nnparams']['learning_rate'])
batch_size = int(config['nnparams']['batch_size'])
num_epochs = int(config['nnparams']['num_epochs'])


train_dataset = datasets.MNIST(root='../datasets/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='../datasets/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = model(input_size=input_size, n_classes=n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_id, (data, target) in enumerate(train_loader):
        data = data.to(device=device)
        targets = target.to(device=device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


print(f"Accuracy on training set: {accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {accuracy(test_loader, model)*100:.2f}")