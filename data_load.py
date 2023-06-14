from tensorflow.keras.preprocessing.image import ImageDataGenerator

class HanjaData:
    def get_data(RawDataPath):
        # Set the desired image size
        ImageSize = (72, 72)

        # Set the desired split ratio
        SplitRatio = 0.2  # 20% for testing

        # Create the data generator with validation split
        # Load the image data using the data generator
        DataGenerator = ImageDataGenerator(
            rescale=1./255,                 # 이미지 픽셀 0~1로 조정
            validation_split=SplitRatio,    # 훈련데이터 0.8 / 검증데이터 0.2
            rotation_range=10,              # 이미지를 10도 범위 내에서 회전
            width_shift_range=0.1,          # (가로의 0.1 범위 내에서) 가로 방향으로 무작위 이동
            height_shift_range=0.1,         # (세로의 0.1 범위 내에서) 세로 방향으로 무작위 이동
            zoom_range=0.1,                 # 0.1배 무작위 확대/축소
            horizontal_flip=False           # 수평뒤집기 x
        )


        TrainGenerator = DataGenerator.flow_from_directory(
                                                            RawDataPath,
                                                            target_size=ImageSize,
                                                            batch_size=BatchSize,
                                                            class_mode='categorical',    # 생성할 레이블 유형. 레이블이 원핫인코딩됨
                                                            shuffle=True,                # 매 에포트 마다 이미지 순서 섞기
                                                            subset='training'            # 이 생성기는 학습데이터로 사용할 것이다!
                                                            )

        ValGenerator = DataGenerator.flow_from_directory(
                                                        RawDataPath,
                                                        target_size=ImageSize,
                                                        batch_size=BatchSize,
                                                        class_mode='categorical',
                                                        shuffle=True,
                                                        subset='validation'           # 이 생성기는 검증데이터로 사용할 것이다!
                                                        )
    
        return TrainGenerator, ValGenerator








