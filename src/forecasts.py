import pandas as pd
from prophet import Prophet

def train_prophet_model(df, column):
    m = Prophet()
    m.fit(df[['Timestamp', column]].rename(columns={'Timestamp': 'ds', column: 'y'}))
    
    last_timestamp = df['Timestamp'].max()
    future_48h = pd.date_range(start=last_timestamp, periods=49, freq='H')[1:]
    future_48h_df = pd.DataFrame({'ds': future_48h})
    
    forecast_48h = m.predict(future_48h_df)
    
    return forecast_48h[['ds', 'yhat']]

def main():
    input_file = "../static/SensorMLDataset_small.csv"
    output_file = "../static/forecasts.csv"
    
    df = pd.read_csv(input_file)
    df.head()

    columns_to_predict = ['pres', 'temp1', 'umid', 'temp2', 'V450', 'B500', 'G550', 'Y570', 'O600', 'R650', 'temps1', 'temps2', 'lumina']

    all_forecasts = pd.DataFrame()

    for column in columns_to_predict:
        forecast = train_prophet_model(df, column)
        all_forecasts[column] = forecast['yhat']

    all_forecasts.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
