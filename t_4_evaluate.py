import tensorflow as tf
import logging


class OCREvaluate:
    def test_eval(test_x, test_y):
        test_x = test_x / 255.0

        model = tf.keras.models.load_model(model_path)

        loss, acc = model.evaluate(test_x, test_y)
        print(f"-----model-----\nloss: {loss:.4f} acc: {acc:.4f}")


    # def OCRLog():
    #     logger = logging.getLogger("airflow-mnist")

    #     logger.setLevel(logging.INFO)

    #     stream_handler = logging.StreamHandler()
    #     file_handler = logging.FileHandler(log_path)

    #     formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    #     stream_handler.setFormatter(formatter)
    #     file_handler.setFormatter(formatter)

    #     logger.addHandler(stream_handler)
    #     logger.addHandler(file_handler)

    #     logger.info(f"model, {model_path}")
    #     logger.info(f"loss, {loss}")
    #     logger.info(f"acc, {acc}")


