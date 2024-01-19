import json

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, optim

from src.lstm.lstm import LSTM


class LSTMService:
    """
    This class is a wrapper for the LSTM class.

    It is used to train the model and to predict the values of the data.
    """
    hidden_size = 64
    num_layers = 2
    seq_len = 10

    def __init__(self, data: pd.DataFrame):
        """
        :param data: pandas DataFrame

        The data should be a pandas DataFrame with the following format:
        - The first column should be the date (it will be ignored)
        - The remaining columns should be the attributes
        """
        data = data.drop("Timestamp", axis=1)
        self.columns = data.columns.values.tolist()
        data = (data - data.mean()) / data.std()
        self.data = data.to_numpy()
        input_size = self.data.shape[1]
        output_size = input_size
        self.model = LSTM(input_size, self.hidden_size, self.num_layers, output_size)

    def train(self, num_epochs, batch_size):
        """
        :param num_epochs: number of epochs
        :param batch_size: batch size

        :return: json array of train losses (length = num_epochs)

        Train the model on the data
        """
        X, y = [], []

        for i in range(len(self.data) - self.seq_len):
            X.append(self.data[i:i + self.seq_len, :])  # change to include all attributes
            y.append(self.data[i + self.seq_len, :])  # change to include all attributes

        X = np.array(X)
        y = np.array(y)
        X_train = torch.from_numpy(X).float()
        y_train = torch.from_numpy(y).float()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        train_losses = []
        min_loss = np.inf
        epochs_no_improve = 0
        n_epochs_stop = 5
        early_stop_threshold = 0.0001

        for epoch in range(num_epochs):
            for i in range(0, X_train.size(0), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

            if loss.item() < min_loss - early_stop_threshold:
                epochs_no_improve = 0
                min_loss = loss.item()
            else:
                epochs_no_improve += 1
                if epochs_no_improve == n_epochs_stop:
                    print(f"Early stopping at epoch {epoch + 1}, Train Loss: {loss.item():.4f}")
                    break

            print(f"Epoch {epoch + 1}, Train Loss: {loss.item():.4f}")
            train_losses.append(loss.item())

        return json.dumps(train_losses)

    def predict(self, data: pd.DataFrame):
        """
        :param data: pandas DataFrame

        :return: y_test, y_pred_test

        Predict the values of the data

        The data should be a pandas DataFrame with the following format:
        - The first column should be the date (it will be ignored)
        - The remaining columns should be the attributes
        """
        data = data.drop("Timestamp", axis=1)
        data = (data - data.mean()) / data.std()
        data = data.to_numpy()
        X_test, y_test = [], []

        for i in range(len(data) - self.seq_len):
            X_test.append(data[i:i + self.seq_len, :])  # change to include all attributes
            y_test.append(data[i + self.seq_len, :])  # change to include all attributes

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()

        self.model.eval()
        with torch.no_grad():
            y_pred_test = self.model(X_test)

        results = dict()

        for i in range(len(self.columns)):
            results[self.columns[i]] = {"actual": y_test[:, i].tolist(), "predicted": y_pred_test[:, i].tolist()}

        return results

    @staticmethod
    def plot_losses(train_losses):
        plt.figure(figsize=(8, 5))
        plt.title("Training loss")
        plt.plot(train_losses, label="Train")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_predictions(y_pred_test, y_test):
        plt.figure(figsize=(8, 5))
        plt.title(f"Predicted vs Actual")
        plt.plot(y_test, label="Actual")
        plt.plot(y_pred_test, label="Predicted")
        plt.xlabel("Time step")
        plt.ylabel("Normalized value")
        plt.legend()
        plt.show()


# Example usage
data = pd.read_csv("../../static/SensorMLTestDataset.csv")
lstm_service = LSTMService(data)
json_train_losses = lstm_service.train(num_epochs=100, batch_size=64)
train_losses = json.loads(json_train_losses)
lstm_service.plot_losses(train_losses)

test_data = pd.read_csv("../../static/SensorMLTestDataset.csv")
results = lstm_service.predict(test_data)
y_test = np.array(results["lumina"]["actual"])
y_pred_test = np.array(results["lumina"]["predicted"])
lstm_service.plot_predictions(y_pred_test, y_test)
