import os
import pathlib


class PathConfig:
    def __init__(self):
        self.project_path = pathlib.Path(__file__).parent.resolve()
        self.data_path = f"{self.project_path}/data"
        self.file_name = 'hanja_data'
        self.train_frac = 0.7
        self.random_num = 2
        self.num = 28
        self.epochs_num = 10


