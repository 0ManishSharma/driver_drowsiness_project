import os
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense,Input,GlobalAveragePooling2D,Dropout
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam
from dataclasses import dataclass
import numpy as np


@dataclass
class ModelTrainerConfig:
    model_dir = "artifacts/model"
    model_name = "driver_drowsiness_model.keras"
    epochs :int  = 20
    lr : float = 1e-4

class ModelTrainer:
    def __init__(self):
        self.config= ModelTrainerConfig()
        os.makedirs(self.config.model_dir,exist_ok=True)

    def build_model(self):
        base_model = EfficientNetB0(

            input_tensor=Input(shape=(224, 224, 3)),
            include_top=False,
            weights="imagenet"

        )
        for layer in base_model.layers:
            layer.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.4)(x)
        x = Dense(256,activation="relu")(x)
        output = Dense(4,activation="softmax")(x)

        model = Model(inputs=base_model.input, outputs=output)

        model.compile(
            optimizer = Adam(self.config.lr),
            loss = "sparse_categorical_crossentropy",
            metrics = ['accuracy']
        )

        return model
    def train(self):
        X_train = np.load("artifacts/data_split/X_train.npy")
        X_val   = np.load("artifacts/data_split/X_val.npy")
        y_train = np.load("artifacts/data_split/y_train.npy")
        y_val   = np.load("artifacts/data_split/y_val.npy")
        model = self.build_model()

        print("🚀 EfficientNetB3 Training Started (GPU)")

        model.fit(
            X_train,
            y_train,
            validation_data = (X_val,y_val),
            epochs = self.config.epochs,
            batch_size=32

        )
        model.save(os.path.join(self.config.model_dir,self.config.model_name))

        print("Model successfully saved")
        


