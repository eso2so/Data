import pandas as pd

from TestProject.t_1_datasets import HanjaData
from TestProject.t_2_train import OCRModeling
from TestProject.t_4_evaluate import OCREvaluate
from TestProject.t_5_config import PathConfig

class OCRMain:
    def __init__(self):
        HanjaData.__init__(self)
        OCRModeling.__init__(self)
        OCREvaluate.__init__(self)
        PathConfig.__init__(self)

    def ready_data(self):
        hanja_data = self.get_data(self.data_path, self.file_name, flag)
        X_train, X_val, X_test,  Y_train, Y_val, Y_test = \
            self.X_Y_split(hanja_data['images'], hanja_data['labels'], self.train_frac, self.random_num)        

    def run_eval_data():
        self.X_train
        self.test_eval




