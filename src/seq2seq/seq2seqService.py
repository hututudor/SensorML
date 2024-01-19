import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
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

input_seq = Input(shape=(n_steps_in, len(columns_to_predict)))
encoder_lstm = LSTM(64, activation='tanh', return_state=True)
_, state_h, state_c = encoder_lstm(input_seq)

decoder_input = RepeatVector(n_steps_out)(state_h)
decoder_lstm = LSTM(64, activation='tanh', return_sequences=True)(decoder_input)
output = TimeDistributed(Dense(len(columns_to_predict)))(decoder_lstm)

model = Model(inputs=input_seq, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

new_data = np.array([test_data[:n_steps_in]])
new_predictions = model.predict(new_data)

predicted_values_original = scaler.inverse_transform(new_predictions.reshape(-1, len(columns_to_predict)))
test_data_original = scaler.inverse_transform(test_data)

for i, col in enumerate(columns_to_predict):
    plt.figure(figsize=(12, 4))

    prediction_times = pd.date_range(start=df.index[-n_steps_in], periods=n_steps_out+1, freq='H')[1:]
    plt.plot(prediction_times, test_data_original[-n_steps_out:, i], label=f'Actual {col}', linestyle='-', marker='o')
    plt.plot(prediction_times, predicted_values_original[:, i], label=f'Predicted {col}', linestyle='--', marker='o')

    plt.title(f'Evoluția valorii {col} în ultima săptămână')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
