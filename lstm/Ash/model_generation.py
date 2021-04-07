import torch
from LSTMSoil import LSTMSoil

 # important area to tweak!
epochs = 100 # number of epochs
learning_rate = 0.001 # learning rate
hidden_size = 10 # number of neurons in the hidden state
n_layers = 1 # number of stacked lstm layers
output_size = 1 # output a real number

def get_model(input_size: int, device: torch.device) -> LSTMSoil:
    model = LSTMSoil(input_size, hidden_size, output_size, n_layers, 1, device)
    model.to(device)
    return model

def get_model_hidden_size_estimation(input_size: int, device: torch.device, hidden_size: int):
    model = LSTMSoil(input_size, hidden_size, output_size, n_layers, 1, device)
    model.to(device)
    return model