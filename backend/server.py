from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sys
sys.path.append('..')
from src.lstm.lstm_service import LSTMService
from src.prophet.prophet_service import ProphetService
from src.seq2seq.seq2seq_service import Seq2SeqService
from flask import send_from_directory

EPOCHS = 100

diseases = {
    "early_blight": {
        "temp": [24, 29],
        "umid": [90, 100]
    },
    "gray_mold": {
        "temp": [17, 23],
        "umid": [90, 100]
    },
    "late_blight": {
        "temp": [10, 24],
        "umid": [90, 100]
    },
    "leaf_mold": {
        "temp": [21, 24],
        "umid": [85, 100]
    },
    "powdery_mildew": {
        "temp": [22, 30],
        "umid": [50, 75]
    },
}


results_store = {}
app = Flask(__name__)
CORS(app)

lstm_service = None
prophet_service = None
seq2seq_service = None

result = {}

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('../static/predictions_plots', filename)

@app.route('/data', methods=['POST'])
def predict_data():
    global lstm_service
    global prophet_service
    global seq2seq_service

    file = request.files['file']
    test_data = pd.read_csv(file)

    # LSTM
    y_pred_test, y_test = lstm_service.predict(test_data)
    data_mean = test_data.drop("Timestamp", axis=1).mean().to_numpy()
    data_std = test_data.drop("Timestamp", axis=1).std().to_numpy()
    lstm_service.plot_predictions(y_pred_test, y_test, data_mean, data_std, path="../static/predictions_plots/lstm")

    lstm_risks = LSTMService.calculate_disease_risk(y_pred_test, data_mean, data_std, diseases)

    # Prophet
    prophet_service.plot_predictions(test_data)

    # Seq2Seq
    service.predict(test_data)
    service.generate_plots()

    global result
    result = {
        "lstm": lstm_risks
    }

    return jsonify({ })



@app.route('/data', methods=['GET'])
def get_data():
    global result
    return jsonify(result)



if __name__ == '__main__':
    data = pd.read_csv("../static/SensorMLTrainDataset.csv")
    service = Seq2SeqService(data)
    service.train(epochs=EPOCHS, batch_size=32)

    data = pd.read_csv("../static/SensorMLTrainDataset.csv")
    lstm_service = LSTMService(data)
    train_losses = lstm_service.train(num_epochs=EPOCHS, batch_size=64)

    data = pd.read_csv("../static/SensorMLTrainDataset.csv")
    prophet_service = ProphetService(data)

    app.run(debug=True)