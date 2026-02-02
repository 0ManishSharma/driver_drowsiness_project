import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    img_size : int = 224
    epochs : int =10
    batch_size : int = 16
    model_dir : str =   "artifacts/model"
    model_name : str = "drowsiness_mdoel.keras"


def build_model(img_size):
    model = Sequential()
    model.add(Conv2D(32,3,activation='relu',input_shape=(img_size,img_size,3)))
    model.add(MaxPooling2D())

    model.add(Conv2D(64,3,activation="relu"))
    model.add(MaxPooling2D())

    model.add(Conv2D(128,3,activation="relu"))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(4,activation="sigmoid"))


    return model

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        os.makedirs(self.config.model_dir,exist_ok=True)

    def train(self):
        # Load Transformed Data
        X_train = np.load("artifacts/data_split/X_train.npy")
        X_val = np.load("artifacts/data_split/X_val.npy")
        y_train = np.load("artifacts/data_split/y_train.npy")
        y_val  = np.load("artifacts/data_split/y_val.npy")

        model = build_model(self.config.img_size)

        model.compile(
            optimizer = "adam",
            loss = "sparse_categorical_crossentropy",
            metrics = ['accuracy']

        )

        print("model Training Started")

        model.fit(
            X_train,
            y_train,
            validation_data = (X_val,y_val),
            epochs = self.config.epochs,
            batch_size = self.config.batch_size

        )
        model.save(os.path.join(self.config.model_dir,self.config.model_name))

        print("✅ Model Training Completed")
        print("📁 Model saved in artifacts/model/")

