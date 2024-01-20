import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, TimeDistributed
from tensorflow.keras.layers import MultiHeadAttention  # Import direct din TensorFlow
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

from src.extractor import extract_dataset

data = extract_dataset()

columns_to_predict = ['pres', 'temp1', 'umid', 'temp2', 'V450', 'B500', 'G550', 'Y570', 'O600', 'R650', 'temps1',
                      'temps2', 'lumina']
df = pd.DataFrame(data[1:], columns=['Timestamp'] + columns_to_predict)

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df[columns_to_predict] = df[columns_to_predict].apply(pd.to_numeric)
df.set_index('Timestamp', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

def create_sequences(data, n_steps_in, n_steps_out):
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

n_steps_in = 168
n_steps_out = 168

train_data = scaled_data[:-n_steps_out]
test_data = scaled_data[-n_steps_out:]

X_train, y_train = create_sequences(train_data, n_steps_in, n_steps_out)

# Define the input layer
input_seq = Input(shape=(n_steps_in, len(columns_to_predict)))

# Transformer Encoder
encoder_output = MultiHeadAttention(num_heads=2, key_dim=len(columns_to_predict)//2)(query=input_seq, value=input_seq)
encoder_output = Dropout(0.1)(encoder_output)
encoder_output = LayerNormalization(epsilon=1e-6)(input_seq + encoder_output)

# Transformer Decoder
decoder_output = MultiHeadAttention(num_heads=2, key_dim=len(columns_to_predict)//2)(query=encoder_output, value=encoder_output)
decoder_output = Dropout(0.1)(decoder_output)
decoder_output = LayerNormalization(epsilon=1e-6)(encoder_output + decoder_output)

# Output layer
output = TimeDistributed(Dense(len(columns_to_predict)))(decoder_output)

model = Model(inputs=input_seq, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

new_predictions = model.predict(train_data[-n_steps_in:].reshape(1, n_steps_in, len(columns_to_predict)))

predicted_values_original = scaler.inverse_transform(new_predictions.reshape(-1, len(columns_to_predict)))
test_data_original = scaler.inverse_transform(test_data)

for i, col in enumerate(columns_to_predict):
    plt.figure(figsize=(12, 4))

    prediction_times = pd.date_range(start=df.index[-n_steps_in], periods=n_steps_out+1, freq='H')[1:]
    plt.plot(prediction_times, test_data_original[:, i], label=f'Actual {col}', linestyle='-', marker='')
    plt.plot(prediction_times, predicted_values_original[:, i], label=f'Predicted {col}', linestyle='-', marker='')

    plt.title(f'Evoluția valorii {col} în ultima săptămână')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
