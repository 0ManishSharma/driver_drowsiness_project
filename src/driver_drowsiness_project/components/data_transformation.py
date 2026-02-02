import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    ingestion_dir: str = "artifacts/data_ingestion"
    split_dir: str = "artifacts/data_split"
    test_size: float = 0.2
    random_state: int = 42


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        os.makedirs(self.config.split_dir, exist_ok=True)

    def transform(self):
        labels = np.load(
            os.path.join(self.config.ingestion_dir, "labels.npy"),
            allow_pickle=True
        )

        X, y = [], []

        for img_path, label in labels:
            img = cv2.imread(img_path)
            img = img / 255.0
            X.append(img)
            y.append(label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config.test_size,
            stratify=y,
            random_state=self.config.random_state
        )

        np.save(os.path.join(self.config.split_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.config.split_dir, "X_val.npy"), X_val)
        np.save(os.path.join(self.config.split_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.config.split_dir, "y_val.npy"), y_val)

        print("✅ Data Transformation Completed")
