import numpy as np
import tensorflow as tf

class ModelEvaluation:
    def evaluate(self):
        model = tf.keras.models.load_model("artifacts/model/driver_drowsiness_model.keras")

        X_val = np.load("artifacts/data_split/X_val.npy")
        y_val = np.load("artifacts/data_split/y_val.npy")

        loss, accuracy = model.evaluate(X_val, y_val)

        print(f"📊 Validation Accuracy: {accuracy:.4f}")
        print(f"📉 Validation Loss: {loss:.4f}")
