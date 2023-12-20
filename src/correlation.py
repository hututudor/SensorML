import numpy as np
from datetime import datetime


def correlation_matrix(dataset: list[list]) -> np.ndarray:
    updated_dataset = _convert_timestamp(dataset)

    transposed = np.transpose(updated_dataset[1:])
    correlation_matrix = np.corrcoef(transposed.astype(float), rowvar=True)
    return correlation_matrix


def _convert_timestamp(dataset: list[list]) -> list[list]:
    """Converts the timestamp column to a float representing the number of seconds since the epoch (1970-01-01)"""
    for row in dataset[1:]:
        timestamp_obj = datetime.strptime(row[0], "%m/%d/%Y %H:%M")
        row[0] = timestamp_obj.timestamp()

    return dataset
