import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    TimeDistributed,
)
from tensorflow.keras.layers import MultiHeadAttention  # Import direct din TensorFlow
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go

from src.extractor import extract_dataset


class Seq2SeqService:
    """
    Service for constructing, training and generating plots for the Seq2Seq model
    """

    def __init__(self):
        self.data = extract_dataset()
        self.path = "../../static/predictions_plots/seq2seq"

        self.n_steps_in = 168
        self.n_steps_out = 168

        self.columns_to_predict = [
            "pres",
            "temp1",
            "umid",
            "temp2",
            "V450",
            "B500",
            "G550",
            "Y570",
            "O600",
            "R650",
            "temps1",
            "temps2",
            "lumina",
        ]
        self.df = pd.DataFrame(
            data[1:], columns=["Timestamp"] + self.columns_to_predict
        )

        self.df["Timestamp"] = pd.to_datetime(self.df["Timestamp"])
        self.df[self.columns_to_predict] = self.df[self.columns_to_predict].apply(
            pd.to_numeric
        )
        self.df.set_index("Timestamp", inplace=True)

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(self.df)

        self.train_data = self.scaled_data[: -self.n_steps_out]
        self.test_data = self.scaled_data[-self.n_steps_out :]

        self.X_train, self.y_train = self._create_sequences(
            self.train_data, self.n_steps_in, self.n_steps_out
        )

        # Define the input layer
        self.input_seq = Input(shape=(self.n_steps_in, len(self.columns_to_predict)))

        # Transformer Encoder
        encoder_output = MultiHeadAttention(
            num_heads=2, key_dim=len(self.columns_to_predict) // 2
        )(query=self.input_seq, value=self.input_seq)
        encoder_output = Dropout(0.1)(encoder_output)
        encoder_output = LayerNormalization(epsilon=1e-6)(
            self.input_seq + encoder_output
        )

        # Transformer Decoder
        decoder_output = MultiHeadAttention(
            num_heads=2, key_dim=len(self.columns_to_predict) // 2
        )(query=encoder_output, value=encoder_output)
        decoder_output = Dropout(0.1)(decoder_output)
        decoder_output = LayerNormalization(epsilon=1e-6)(
            encoder_output + decoder_output
        )

        # Output layer
        self.output = TimeDistributed(Dense(len(self.columns_to_predict)))(
            decoder_output
        )

    def _create_sequences(self, data, n_steps_in, n_steps_out):
        X, y = [], []
        for i in range(len(data)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > len(data):
                break
            seq_x, seq_y = data[i:end_ix], data[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def train(self, epochs: int = 100, batch_size: int = 32):
        """
        Trains the model with number of epochs and batch size specified

        Args:
            epochs (int): Number of epochs to train the model
            batch_size (int): Batch size to train the model

        Return: None
        """

        self.model = Model(inputs=self.input_seq, outputs=self.output)
        self.model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")

        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
        )

        new_predictions = self.model.predict(
            self.train_data[-self.n_steps_in :].reshape(
                1, self.n_steps_in, len(self.columns_to_predict)
            )
        )

        self.predicted_values_original = self.scaler.inverse_transform(
            new_predictions.reshape(-1, len(self.columns_to_predict))
        )
        self.test_data_original = self.scaler.inverse_transform(self.test_data)

    def generate_plots(self):
        """
        Generates all the plots for the predicted values

        Return: None
        """
        for i, col in enumerate(self.columns_to_predict):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=self.df.index[-self.n_steps_in :],
                    y=self.df[col].values[-self.n_steps_in :],
                    mode="lines",
                    name=f"Actual {col}",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.df.index[-self.n_steps_in :],
                    y=self.predicted_values_original[:, i],
                    mode="lines",
                    name=f"Predicted {col}",
                )
            )
            fig.update_layout(
                title=f"Predictions for {col}",
                title_x=0.5,
                xaxis_title="Time",
                yaxis_title="Value",
            )
            fig.write_html(f"{self.path}/{col}.html")


"""
Example usage

service = Seq2SeqService()
service.train(epochs=100, batch_size=32)
service.generate_plots()
"""
