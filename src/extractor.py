import os
import csv

__dir__ = os.path.dirname(__file__)


def extract_dataset() -> list[list]:
    dataset_path = os.path.join(__dir__, "../static/SensorMLDataset_small.csv")
    with open(dataset_path, "r", encoding="utf-8-sig") as file:
        dataset = list(csv.reader(file))

    return dataset


print(extract_dataset())
