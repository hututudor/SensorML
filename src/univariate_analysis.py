from extractor import extract_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = extract_dataset()

dataset = extract_dataset()

columns = dataset[0][1:]

for column in columns:
    plt.figure(figsize=(12, 6))

    data_by_column = {row[0]: float(row[columns.index(column) + 1]) for row in dataset[1:]}
    dates = list(data_by_column.keys())
    values = list(data_by_column.values())

    heatmap_data = pd.DataFrame({'data': dates, column: values})
    heatmap_data['data'] = pd.to_datetime(heatmap_data['data'])
    heatmap_data.set_index('data', inplace=True)

    heatmap_data_resampled = heatmap_data.resample('D').agg(['mean', 'median'])
    sns.heatmap(heatmap_data_resampled, cmap='YlGnBu', annot=True, cbar=True, fmt=".2f")
    plt.title(f'Heatmap pentru {column} - Medii și Mediane pe Zi')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=values)
    plt.title(f'Boxplot pentru {column}')
    plt.show()
