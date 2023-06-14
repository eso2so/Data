from tensorflow.keras.applications import VGG16
import tensorflow as tf

class HanjaTrain:
    def run_train(TrainGenerator, ValGenerator, NumClasses, BatchSize, Epochs, model_path):
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(72, 72, 3))

        # Freeze the base model's layers
        base_model.trainable = False

        # Create the model
        model = tf.keras.Sequential([
                                    base_model,
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    tf.keras.layers.Dense(NumClasses, activation='softmax')
                                    ])

        # Define the callbacks
        callbacks = [
                        # monitor='val_accuracy'  ==>  유효성검사 정확도를 모니터링해서
                        # patience=5  ==> 5번 이상 개선되지 않으면 학습 중지
                        # mode='max' ==> 유효성 검사 정확도를 최대화
                        # verbose=1 ==> 교육 진행 및 조기 중지 조건에 대한 자세한 출력 메시지를 생성
                        EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1),
                        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
                    ]

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # model.summary()

        # Train the model with callbacks
        model.fit(
                    TrainGenerator,
                    steps_per_epoch=TrainGenerator.samples // BatchSize,
                    validation_data=ValGenerator,
                    validation_steps=ValGenerator.samples // BatchSize,
                    epochs=Epochs,
                    callbacks=callbacks
                )
        
        model.save(model_path)









