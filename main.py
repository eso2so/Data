from ocr.data_load import HanjaData
from ocr.train import HanjaTrain
from ocr.config import PathConfig

class OCRMain:
    def __init__(self):
        HanjaData.__init__(self)
        HanjaTrain.__init__(self)
        PathConfig.__init__(self)

    def ready_data(self):
        TrainGenerator, ValGenerator = self.get_data(self.data_path)
 
    def train_eval_data(self):
        run_train(TrainGenerator, ValGenerator, self.NumClasses, self.BatchSize, self.Epochs, self.model_path)
