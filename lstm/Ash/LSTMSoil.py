import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTMSoil(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, seq_len, device: torch.device):
        super(LSTMSoil, self).__init__()
        # setup neural network params
        self.input_size = input_size # number of features
        self.hidden_size = hidden_size # hidden layer size (# neurons for hidden state)
        self.output_size = output_size # output size, should just be 1 since we're doing linear regression
        self.n_layers = n_layers # number of layers for the LSTM, usually 2, hyperparameter
        self.seq_len = seq_len # sequence length
        self.device = device
        self.path = None

        # setup neural network layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True) # lstm layer
        # self.fc_1 = nn.Linear(hidden_size, 128) # fully connected layer, could be useful for better accuracy, needs testing
        self.fc = nn.Linear(hidden_size, output_size) # fully connected output layer

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_size)).to(self.device) # set initial hidden state
        c_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_size)).to(self.device) # set initial cell state
        # send input through LSTM
        _, (hn, _) = self.lstm(x, (h_0, c_0)) # lstm with input x, hidden tuple
        # reshape the hidden cell for the fully connected layer
        hn = hn.view(-1, self.hidden_size)
        out = self.fc(hn)
        return out

    # to update path
    def update_path(self, path: str):
        self.path = path
    
    # for string representation
    def __str__(self):
        st, f = "(lstm): " + self.lstm.__str__(), "(fc): " + self.fc.__str__()
        if self.path is not None:
            return "LSTMSoil(\n  {}\n  {}\n  (path): {}\n)".format(st, f, self.path)
        else:
            return "LSTMSoil(\n  {}\n  {}\n)".format(st, f)
        
