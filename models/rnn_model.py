import torch
import torch.nn as nn
import torch.nn.functional as _f

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, sequence_length, n_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embeddings = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, n_classes)


    def forward(self, x, hidden, cell):
        x = self.embeddings(x)
        x, (hidden, cell) = self.lstm(x.unsqueeze(1), (hidden,cell))
        x = self.fc(x.reshape(x.shape[0], -1))

        return x, (hidden, cell)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell
        