import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet


class ProphetService:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __train(self, column):
        self.model = Prophet()
        self.model.fit(self.data.rename(columns={'Timestamp': 'ds', column: 'y'}))

    def predict(self):
        last_timestamp = self.data['Timestamp'].max()
        additional_df = pd.date_range(start=last_timestamp, periods=168, freq='H')[1:]
        additional_df = pd.DataFrame({'ds': additional_df})

        forecast = self.model.predict(additional_df)

        return forecast

    def plot_predictions(self, test_data: pd.DataFrame):
        last_timestamp = self.data['Timestamp'].max()
        additional_df = pd.date_range(start=last_timestamp, periods=168, freq='H')[1:]
        additional_df = pd.DataFrame({'Timestamp': additional_df})
        for column in self.data.columns[1:]:
            self.__train(column)
            forecast = self.predict()
            figure = go.Figure()
            figure.add_trace(
                dict(x=additional_df['Timestamp'], y=test_data[column], mode='lines', name='Actual values'))
            figure.add_trace(dict(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted values'))
            figure.update_layout(title_text=f'Prophet predictions for {column}', title_x=0.5, xaxis_title="Time",
                                 yaxis_title="Value")
            figure.write_html(f"../../static/predictions_plots/prophet/{column}.html")


# Example usage:
data = pd.read_csv('../../static/SensorMLDataset_small.csv')
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
test_data = pd.read_csv('../../static/SensorMLTestDataset.csv')
prophet = ProphetService(data)
prophet.plot_predictions(test_data)
