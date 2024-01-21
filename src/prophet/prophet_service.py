import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet


class ProphetService:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.data["Timestamp"] = pd.to_datetime(self.data["Timestamp"])

    def __train(self, column):
        self.model = Prophet()
        self.model.fit(self.data.rename(columns={"Timestamp": "ds", column: "y"}))

    def predict(self):
        last_timestamp = self.data["Timestamp"].max()
        additional_df = pd.date_range(start=last_timestamp, periods=168, freq="H")[1:]
        additional_df = pd.DataFrame({"ds": additional_df})

        forecast = self.model.predict(additional_df)

        return forecast

    def plot_predictions(self, test_data: pd.DataFrame):
        self.test_data = test_data
        last_timestamp = self.data["Timestamp"].max()
        additional_df = pd.date_range(start=last_timestamp, periods=168, freq="H")[1:]
        additional_df = pd.DataFrame({"Timestamp": additional_df})
        for column in self.data.columns[1:]:
            self.__train(column)
            forecast = self.predict()
            figure = go.Figure()
            figure.add_trace(
                dict(
                    x=additional_df["Timestamp"],
                    y=test_data[column],
                    mode="lines",
                    name="Actual",
                )
            )
            figure.add_trace(
                dict(
                    x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Predicted"
                )
            )
            figure.update_layout(
                title_text=f"Prophet predictions for {column}",
                title_x=0.5,
                xaxis_title="Time",
                yaxis_title="Value",
            )
            figure.write_html(f"../../static/predictions_plots/prophet/{column}.html")

    def create_disease_risks(self) -> dict:
        self.test_data["Timestamp"] = pd.to_datetime(self.test_data["Timestamp"])
        self.test_data.set_index("Timestamp", inplace=True)
        daily_avg_temp1 = self.test_data["temp1"].resample("D").mean().tolist()
        daily_avg_umid = self.test_data["umid"].resample("D").mean().tolist()

        disease_risks = {}
        diseases = [
            "early_blight",
            "gray_mold",
            "late_blight",
            "leaf_mold",
            "powdery_mildew",
        ]

        temp_intervals = {
            "early_blight": (24, 29),
            "gray_mold": (17, 23),
            "late_blight": (10, 24),
            "leaf_mold": (21, 24),
            "powdery_mildew": (22, 30),
        }

        humidity_intervals = {
            "early_blight": (90, 100),
            "gray_mold": (90, 100),
            "late_blight": (90, 100),
            "leaf_mold": (85, 100),
            "powdery_mildew": (50, 75),
        }

        for disease in diseases:
            temp_interval = temp_intervals[disease]
            humidity_interval = humidity_intervals[disease]

            risk_days = 0
            for temp, humidity in zip(daily_avg_temp1, daily_avg_umid):
                if (
                    temp_interval[0] <= temp <= temp_interval[1]
                    and humidity_interval[0] <= humidity <= humidity_interval[1]
                ):
                    risk_days += 1

            disease_risks[f"{disease}"] = risk_days / len(daily_avg_temp1)

        return disease_risks


# Example usage:
# data = pd.read_csv("../../static/SensorMLDataset_small.csv")
# test_data = pd.read_csv("../../static/SensorMLTestDataset.csv")
# prophet = ProphetService(data)
# prophet.plot_predictions(test_data)
# print(prophet.create_disease_risks())
# If you want to call create_disease_risks() you need to call plot_predictions() first
