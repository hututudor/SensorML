from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sys
sys.path.append('..')
from src.lstm.lstm_service import LSTMService
import uuid
from flask import send_from_directory


results_store = {}
app = Flask(__name__)
CORS(app)

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('plots/predictions', filename)

@app.route('/data', methods=['POST'])
def predict_data():
    file = request.files['file']
    data = pd.read_csv(file)
    lstm_service = LSTMService(data)
    train_losses = lstm_service.train(num_epochs=100, batch_size=64)
    lstm_service.plot_losses(train_losses, path="plots/predictions")
    y_pred_test, y_test = lstm_service.predict(data)
    lstm_service.plot_predictions(y_pred_test, y_test, path="plots/predictions")
    results = {
        'predicted': y_pred_test.tolist(),
        'actual': y_test.tolist()
    }
    id = str(uuid.uuid4())
    results_store[id] = results

    return jsonify({'id': id})

@app.route('/data/<id>', methods=['GET'])
def get_data(id):
    # Check if the ID exists in the store
    if id in results_store:
        return jsonify(results_store[id])
    else:
        return jsonify({'error': 'ID not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)