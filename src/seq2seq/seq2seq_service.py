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
from tensorflow.keras.layers import MultiHeadAttention  
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go


def _create_sequences(data, n_steps_in, n_steps_out):
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


class Seq2SeqService:
    """
    Service for constructing, training and generating plots for the Seq2Seq model
    """

    def __init__(self, data: pd.DataFrame):
        self.predicted_values_original = None
        self.test_data = None
        self.path = "../../static/predictions_plots/seq2seq"

        self.n_steps_in = 168
        self.n_steps_out = 168

        self.columns_to_predict = data.columns[1:]
        self.df = data
        self.df["Timestamp"] = pd.to_datetime(self.df["Timestamp"])
        self.df[self.columns_to_predict] = self.df[self.columns_to_predict].apply(
            pd.to_numeric
        )
        self.df.set_index("Timestamp", inplace=True)

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(self.df)

        self.train_data = self.scaled_data

        self.X_train, self.y_train = _create_sequences(
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

        self.model = Model(inputs=self.input_seq, outputs=self.output)

    def train(self, epochs: int = 100, batch_size: int = 32):
        """
        Trains the model with number of epochs and batch size specified

        Args:
            epochs (int): Number of epochs to train the model
            batch_size (int): Batch size to train the model

        Return: None
        """

        self.model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")

        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
        )

    def predict(self, test_data: pd.DataFrame):
        """
        Predicts the values for the next 168 hours

        Return: None
        """
        self.test_data = test_data
        self.test_data["Timestamp"] = pd.to_datetime(self.test_data["Timestamp"])
        self.test_data.set_index("Timestamp", inplace=True)

        new_predictions = self.model.predict(
            self.train_data[-self.n_steps_in:].reshape(
                1, self.n_steps_in, len(self.columns_to_predict)
            )
        )

        self.predicted_values_original = self.scaler.inverse_transform(
            new_predictions.reshape(-1, len(self.columns_to_predict))
        )

    def generate_plots(self):
        """
        Generates all the plots for the predicted values

        Return: None
        """
        for i, col in enumerate(self.columns_to_predict):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=self.test_data.index,
                    y=self.test_data.loc[:, col],
                    mode="lines",
                    name=f"Actual {col}",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.test_data.index,
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


# Example usage

data = pd.read_csv("../../static/SensorMLDataset_small.csv")
service = Seq2SeqService(data)
service.train(epochs=100, batch_size=32)
test_data = pd.read_csv("../../static/SensorMLTestDataset.csv")
service.predict(test_data)
service.generate_plots()
