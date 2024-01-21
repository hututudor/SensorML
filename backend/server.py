from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sys
sys.path.append('..')
from src.lstm.lstm_service import LSTMService
from src.prophet.prophet_service import ProphetService
from src.seq2seq.seq2seq_service import Seq2SeqService
import uuid
from flask import send_from_directory


results_store = {}
app = Flask(__name__)
CORS(app)



@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('../static/predictions_plots/lstm', filename)

@app.route('/data', methods=['POST'])
def predict_data():
    data = pd.read_csv("../static/SensorMLDataset_small.csv")
    lstm_service = LSTMService(data)
    train_losses = lstm_service.train(num_epochs=100, batch_size=64)
    lstm_service.plot_losses(train_losses, path="../static/predictions_plots/lstm")
    prophet = ProphetService(data)
    file = request.files['file']
    data = pd.read_csv(file)
    #LSTM
    y_pred_test, y_test = lstm_service.predict(data)
    lstm_service.plot_predictions(y_pred_test, y_test, path="../static/predictions_plots/lstm")
    #Prophet
    prophet.plot_predictions(data)
    #Seq2Seq
    service.predict(data)
    service.generate_plots()
    results = {
        'predicted': y_pred_test.tolist(),
        'actual': y_test.tolist()
    }
    id = str(uuid.uuid4())
    results_store[id] = results

    return jsonify({'id': id})



@app.route('/data', methods=['GET'])
def get_data(id):
    # Check if the ID exists in the store
    if id in results_store:
        return jsonify(results_store[id])
    else:
        return jsonify({'error': 'ID not found'}), 404



if __name__ == '__main__':
    data = pd.read_csv("../static/SensorMLDataset_small.csv")
    service = Seq2SeqService(data)
    service.train(epochs=100, batch_size=32)


    app.run(debug=True)