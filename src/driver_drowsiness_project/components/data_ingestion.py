import os
import cv2
import numpy as np
import shutil
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    source_dir: str = "archive/train"
    artifacts_dir: str = "artifacts/data_ingestion"
    img_size: int = 224


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        self.class_map = {
            "Open": 0,
            "Closed": 1,
            "yawn": 2,
            "no_yawn": 3
        }

        self.images_dir = os.path.join(self.config.artifacts_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

    def ingest(self):
        labels = []

        for class_name, label in self.class_map.items():
            src_class_path = os.path.join(self.config.source_dir, class_name)
            dst_class_path = os.path.join(self.images_dir, class_name)

            os.makedirs(dst_class_path, exist_ok=True)

            for img_name in os.listdir(src_class_path):
                if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                src_img = os.path.join(src_class_path, img_name)
                dst_img = os.path.join(dst_class_path, img_name)

                img = cv2.imread(src_img)
                if img is None:
                    continue

                img = cv2.resize(img, (self.config.img_size, self.config.img_size))
                cv2.imwrite(dst_img, img)

                labels.append((dst_img, label))

        labels = np.array(labels, dtype=object)
        np.save(os.path.join(self.config.artifacts_dir, "labels.npy"), labels)

        print("✅ Data Ingestion Completed")
        print("📁 Images stored in artifacts")
        print("🏷️ Labels saved")
