from datetime import datetime

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.plot import plot_plotly

from src.extractor import extract_dataset

DATE_FORMAT = '%m/%d/%Y %H:%M'


def train_prophet_model(df, column):
    model = Prophet()
    model.fit(df[['Timestamp', column]].rename(columns={'Timestamp': 'ds', column: 'y'}))

    return model


def predict_next_48h(model, df):
    last_timestamp = df['Timestamp'].max()
    future_48h = pd.date_range(start=last_timestamp, periods=49, freq='H')[1:]
    future_48h_df = pd.DataFrame({'ds': future_48h})

    forecast_48h = model.predict(future_48h_df)

    return forecast_48h


def get_cross_validation_error(model, df):
    first_timestamp = df['Timestamp'].min()
    last_timestamp = df['Timestamp'].max()

    total_days = (datetime.strptime(last_timestamp, DATE_FORMAT) - datetime.strptime(first_timestamp, DATE_FORMAT)).days

    cv_results = cross_validation(model, horizon='2 days', period='7 days', initial=total_days)

    predictions = cv_results['yhat']
    actual_values = cv_results['y']

    # compute mean absolute error
    return np.mean(np.abs(actual_values - predictions))


def show_figure(fig, title):
    fig.update_layout(title_text=title, title_x=0.5,
                      xaxis=dict(rangeselector=dict(buttons=list([dict(step="all")])), rangeslider=dict(visible=True),
                                 type="date"))
    fig.show()


def main():
    output_file = "../out/forecasts.csv"

    data = extract_dataset()

    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.head(168)

    columns_to_predict = ['pres', 'temp1', 'umid', 'temp2', 'V450', 'B500', 'G550', 'Y570', 'O600', 'R650', 'temps1',
                          'temps2', 'lumina']

    all_forecasts = pd.DataFrame()
    column_cv_results = {}

    figures = []

    for column in columns_to_predict:
        model = train_prophet_model(df, column)

        forecast = predict_next_48h(model, df)
        all_forecasts[column] = forecast[['yhat']]

        figures.append(plot_plotly(model, forecast))
        column_cv_results[column] = get_cross_validation_error(model, df)

    all_forecasts.to_csv(output_file, index=False)

    print("Column CV MAE result:")
    for column in columns_to_predict:
        print(f"{column}: {column_cv_results[column]}")

    for figure in enumerate(figures):
        show_figure(figure[1], columns_to_predict[figure[0]])


if __name__ == "__main__":
    main()
