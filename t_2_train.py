import tensorflow as tf

model_path

class OCRModeling:
    def __init__(self) -> None:
        pass
    
    def run_train(train_x, train_y, num, epochs_num, model_path):
        train_x = train_x / 255.0

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(num, num)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1817, activation='softmax')
            ]
        )
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        model.fit(train_x, train_y, epochs=epochs_num)

        model.save(model_path)
