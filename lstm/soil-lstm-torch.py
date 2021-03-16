import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.autograd import Variable

# load the data
_dir = os.path.abspath('')
data_path = os.path.join(_dir, "../data/daily_cleaned.csv")
df = pd.read_csv(data_path)
df = df.drop(df.columns[0], axis=1)
new_columns = df.columns.values
new_columns[-1] = 'label'
df.columns = new_columns
#print(df)

# split features and label
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

print(X)
print(Y)

# possibly scale dataset here?

# split into train, validation, test sets
# make sure to reshape all the label sets (y-sets) to be [dimension, 1] because of PyTorch
x_train = X.iloc[:1386, :].to_numpy()
y_train = Y.iloc[:1386].to_numpy()
y_train = y_train.reshape((len(y_train), 1))

x_valid = X.iloc[1386:2078, :].to_numpy()
y_valid = Y.iloc[1386:2078].to_numpy()
y_valid = y_valid.reshape((len(y_valid), 1))

x_test = X.iloc[2078:, :].to_numpy()
y_test = Y.iloc[2078:].to_numpy()
y_test = y_test.reshape((len(y_test), 1))

print("Train Shape: ", x_train.shape, y_train.shape)
print("Valid Shape: ", x_valid.shape, y_valid.shape)
print("Test Shape: ", x_test.shape, y_test.shape)

# setup pytorch to use cuda (gpu training) if possible
# pytorch stuff
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# convert datasets to pytorch tensors
x_train_tensors = Variable(torch.Tensor(x_train)).to(device)
y_train_tensors = Variable(torch.Tensor(y_train)).to(device)

x_valid_tensors = Variable(torch.Tensor(x_valid)).to(device)
y_valid_tensors = Variable(torch.Tensor(y_valid)).to(device)

x_test_tensors = Variable(torch.Tensor(x_test)).to(device)
y_test_tensors = Variable(torch.Tensor(y_test)).to(device)

# reshape tensors to [rows, timestamps, features]
# input format for pytorch LSTMs
x_train_tensors = torch.reshape(x_train_tensors, (x_train_tensors.shape[0], 1, x_train_tensors.shape[1]))

x_valid_tensors = torch.reshape(x_valid_tensors, (x_valid_tensors.shape[0], 1, x_valid_tensors.shape[1]))

x_test_tensors = torch.reshape(x_test_tensors, (x_test_tensors.shape[0], 1, x_test_tensors.shape[1]))

print("Train Shape: ", x_train_tensors.shape, y_train_tensors.shape)
print("Valid Shape: ", x_valid_tensors.shape, y_valid_tensors.shape)
print("Test Shape: ", x_test_tensors.shape, y_test_tensors.shape)

# now build the model
class LSTMSoil(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, seq_len):
        super(LSTMSoil, self).__init__()
        # setup neural network params
        self.input_size = input_size # number of features
        self.hidden_size = hidden_size # hidden layer size (# neurons for hidden state)
        self.output_size = output_size # output size, should just be 1 since we're doing linear regression
        self.n_layers = n_layers # number of layers for the LSTM, usually 2, hyperparameter
        self.seq_len = seq_len # sequence length

        # setup neural network layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True) # lstm layer
        # self.fc_1 = nn.Linear(hidden_size, 128) # fully connected layer, could be useful for better accuracy, needs testing
        self.fc = nn.Linear(hidden_size, output_size) # fully connected output layer
    
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_size)).to(device) # set initial hidden state
        c_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_size)).to(device) # set initial cell state
        # send input through LSTM
        lstm_out, (hn, cn) = self.lstm(x, (h_0, c_0)) # lstm with input x, hidden tuple
        # reshape the hidden cell for the fully connected layer
        hn = hn.view(-1, self.hidden_size)
        out = self.fc(hn)
        return out

# define hyperparameter variables
# important area to tweak!
epochs = 100 # number of epochs
learning_rate = 0.001 # learning rate
input_size = 4 # number of features
hidden_size = 50 # number of neurons in the hidden state
n_layers = 1 # number of stacked lstm layers
output_size = 1 # output a real number

# instantiate the model
model = LSTMSoil(input_size, hidden_size, output_size, n_layers, x_train_tensors.shape[1]) # LSTM model class
model.cuda()
print(model) # print the model

# define loss function and optimizer
# also hyperparameters
loss_fn = torch.nn.MSELoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate)

# setup stats collection
agg_train_loss = [] # training loss over time
agg_valid_loss = [] # validation set loss over time

# train the model
model.train()
for epoch in range(epochs):
    outputs = model(x_train_tensors) # pass inputs through model
    optimizer.zero_grad() # calculate gradient, manually setting it to 0
    # calculate loss
    loss = loss_fn(outputs, y_train_tensors)
    loss.backward()
    optimizer.step() # backpropagation

    agg_train_loss.append(loss.item())

    model.eval() # set model to be in evaluation mode
    val_outs = model(x_valid_tensors)
    val_loss = loss_fn(val_outs, y_valid_tensors)
    agg_valid_loss.append(val_loss.item())
    model.train() # set model to be in training mode

    print("Epoch: %d, Loss: %1.5f, Validation Loss: %1.5f" % (epoch, loss.item(), val_loss.item()))

# now run model on test dataset
model.eval()
y_pred = model(x_test_tensors)
test_loss = loss_fn(y_pred, y_test_tensors)
print("Test Loss: %1.5f" % (test_loss.item()))

# plot validation loss and training loss
plt.plot(agg_train_loss, label='train')
plt.plot(agg_valid_loss, label='validation')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()