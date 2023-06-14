import os
import pathlib


class PathConfig:
    def __init__(self):
        self.project_path = pathlib.Path(__file__).parent.resolve()
        self.data_path = f"{self.project_path}/hanja_data"
        self.model_path = f"{self.project_path}/data"
        self.NumClasses = 1817
        self.Epochs = 150
        self.BatchSize = 8