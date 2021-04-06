import os
import torch
import torch.nn as nn
import torch.tensor as Tensor
from torch.optim import Adam
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.autograd import Variable
from LSTMSoil import LSTMSoil
import io
from types import FunctionType

# load the data
_dir = os.path.abspath('')
output_column = 'label'

def get_data_path() -> str:
    #original data  
    #path = "../../data/daily_cleaned.csv"
    #modified data
    path = "../../data/modified_data/modified_data.csv"
    return path

def get_data(path: str) -> pd.DataFrame:
    data_path = os.path.join(_dir, path)    
    df = pd.read_csv(data_path)
    df = df.drop(df.columns[0], axis=1)

    new_columns = df.columns.values
    new_columns[-1] = output_column
    df.columns = new_columns
    return df

# possibly scale dataset here?
def split_sequence(dataframe: pd.DataFrame, outputColName: str, steps: int):
    
    for i in range(steps, 0, -1):
        kwargs = {'{}(t-{})'.format(outputColName,i): dataframe[outputColName].shift(i).values}
        dataframe = dataframe.assign(**kwargs)
    dataframe = dataframe.fillna(0)
    return dataframe

# get features as X and output as Y
def get_features_op(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # splitting the sequence to get historical output values to use in the lstm
    df = split_sequence(dataframe=df, outputColName=output_column, steps=10)
    # split features and label
    X = df.drop(labels=output_column, axis=1)
    Y = df[output_column]
    # print(X, Y, sep='\n')
    return X, Y

# plot validation loss and training loss
def plot_losses(train_losses: list, val_losses: list, xlabel: str, ylabel: str):
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='validation')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# split data into x_train, x_val, x_test and y_train, y_val, y_test
def split_data(X: pd.DataFrame, Y: pd.DataFrame) -> (np.array, np.array, np.array, np.array, np.array, np.array) :
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

    # print("Train Shape: ", x_train.shape, y_train.shape)
    # print("Valid Shape: ", x_valid.shape, y_valid.shape)
    # print("Test Shape: ", x_test.shape, y_test.shape)
    return x_train, x_valid, x_test, y_train, y_valid, y_test

# setup pytorch to use cuda (gpu training) if possible
def setup_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device

def convert_to_tensor(arr: np.array, device: torch.device, reshape: bool = True) -> Tensor :
    tensor = Variable(torch.Tensor(arr)).to(device)
    if reshape: tensor = torch.reshape(tensor, (tensor.shape[0], 1, tensor.shape[1]))
    return tensor

def get_trained_model(input_size: int, hidden_size: int, output_size: int, n_layers: int
                        , x_train_tensor: Tensor, device: torch.device) -> LSTMSoil:
    model = LSTMSoil(input_size, hidden_size, output_size, n_layers, x_train_tensor.shape[1], device=device) # LSTM model class
    model.to(device)
    model.train()
    return model

# break tensor data to training and validation tensors according to the fraction provided
def get_train_val_sets(data: Tensor, frac: float = 0.6) -> (Tensor, Tensor):
    end = round(float(len(data)) * frac)
    return data[0:end], data[end: len(data)]
    
#k-fold cross validation
def perform_kfold_cross_val(x_train: Tensor, y_train: Tensor, optimizer: Adam
                                , model: LSTMSoil, folds: int, loss_fn: nn.MSELoss) -> (list, list):
    fold_increment = round((len(x_train) * 1.0)/float(folds)) + 1
    # print(len(x_train))
    fold_end = 0
    train_losses, val_losses = list(), list()
    for fold in range(folds):
        optimizer.zero_grad()
        fold_end += fold_increment
        if fold_end >= len(x_train): fold_end = len(x_train)

        (x_fold_train, x_fold_val), (y_fold_train, y_fold_val) = get_train_val_sets(data=x_train[0: fold_end]), get_train_val_sets(data=y_train[0:fold_end])
        output = model(x_fold_train)
        loss = loss_fn(output, y_fold_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        model.eval()
        val_outs = model(x_fold_val)
        val_loss = loss_fn(val_outs, y_fold_val)
        val_losses.append(val_loss.item())
        model.train()
        # print("Fold: {}, Loss: {}, Validation Loss: {}, Row End: {}".format(fold + 1, loss.item(), val_loss.item(), fold_end))
    return train_losses, val_losses

#hyperparameter estimation
def test_cross_validation(x_train_tensors: Tensor, y_train_tensors: Tensor, optimizer: Adam
                            , model: LSTMSoil, loss_fn: nn.MSELoss, max_folds: int) -> (list, list, list):
    agg_train_loss, agg_valid_loss = list(), list()
    for i in range(max_folds):
        train_loss, val_loss = perform_kfold_cross_val(x_train_tensors, y_train_tensors, optimizer, model, i + 1, loss_fn)
        agg_train_loss.append(np.mean(train_loss))
        agg_valid_loss.append(np.mean(val_loss))
        print("K-Folds: {}, Mean Train Loss: {}, Mean Validation Loss: {}".format(i + 1, agg_train_loss[-1], agg_valid_loss[-1]))
    return agg_train_loss, agg_valid_loss

def run_epochs(model: LSTMSoil, x_train_tensors: Tensor, y_train_tensors: Tensor, x_valid_tensors: Tensor
                , y_valid_tensors: Tensor, optimizer: Adam, loss_fn: nn.MSELoss, epochs: int): 
    agg_train_loss, agg_valid_loss = list(), list()
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
    print("Mean Training Loss: {}, Mean Validation Loss: {}".format(np.mean(agg_train_loss), np.mean(agg_valid_loss)))

def test_result(model: LSTMSoil, x_test_tensor: Tensor, y_test_tensor: Tensor, loss_fn: nn.MSELoss):
    # now run model on test dataset 
    model.eval()
    y_pred = model(x_test_tensor)
    test_loss = loss_fn(y_pred, y_test_tensor)
    print("Test Loss: %1.5f" % (test_loss.item()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ld', '--load', type=str, help='Model file to load (state dict)') # option to load a saved model
    parser.add_argument('-sv', '--save', default=0, type=int, help='Save model or not (1 for true, 0 for false)')
    args = parser.parse_args()

    path = get_data_path()
    df: pd.DataFrame = get_data(path)
    X, Y = get_features_op(df)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(X, Y)
    device = setup_device()
    x_train_tensor, x_test_tensor = convert_to_tensor(x_train, device), convert_to_tensor(x_test, device)
    y_train_tensor, y_test_tensor = convert_to_tensor(y_train, device, reshape=False), convert_to_tensor(y_test, device, reshape=False)

    # define hyperparameter variables
    # important area to tweak!
    epochs = 100 # number of epochs
    learning_rate = 0.001 # learning rate
    input_size = len(X.columns) # number of features
    hidden_size = 50 # number of neurons in the hidden state
    n_layers = 1 # number of stacked lstm layers
    output_size = 1 # output a real number
    # define loss function and optimizer
    loss_fn = torch.nn.MSELoss()

    # load model parameters if possible
    if args.load is not None:
        # load model weights
        print("Loading model from file:", str(args.load))
        model = LSTMSoil(input_size, hidden_size, output_size, n_layers, x_train_tensor.shape[1], device=device)
        model.load_state_dict(torch.load(args.load, map_location=device))
    else:
        model = get_trained_model(input_size, hidden_size, output_size, n_layers, x_train_tensor, device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # send model to device
    model.to(device)
    train_losses, val_losses = test_cross_validation(x_train_tensor, y_train_tensor, optimizer, model, loss_fn, max_folds=18) # 18 seems best
    min_val_loss = min(val_losses)
    min_val_loss_idx = val_losses.index(min_val_loss)
    print("Min Validation Loss: {} in fold: {}".format(min_val_loss, min_val_loss_idx + 1))
    # plot_losses(np.asarray(train_losses) * 100, np.asarray(val_losses) * 100, xlabel='K', ylabel='Loss x 100')
    
    test_result(model, x_test_tensor, y_test_tensor,loss_fn)

    if args.save == 1:
        # save model
        print("Saving model to ./model/lstm-mode.pt")
        _dir = os.path.dirname(__file__)
        save_local_path = "model/lstm-model.pt"
        save_abs_path = os.path.join(_dir, save_local_path)
        torch.save(model.state_dict(), save_abs_path)
    

if __name__ == "__main__":
    main()