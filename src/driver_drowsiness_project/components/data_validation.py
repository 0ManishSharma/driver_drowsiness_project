import os
import numpy as np
from dataclasses import dataclass

@dataclass
class DataValidationConfig:
    artifacts_dir: str = "artifacts/data_ingestion"
    required_classes = ["Open", "Closed", "yawn", "no_yawn"]


class DataValidation:
    def __init__(self):
        self.config = DataValidationConfig()

    def validate(self):
        images_dir = os.path.join(self.config.artifacts_dir, "images")
        labels_path = os.path.join(self.config.artifacts_dir, "labels.npy")

        if not os.path.exists(labels_path):
            raise Exception("❌ labels.npy not found")

        for cls in self.config.required_classes:
            cls_path = os.path.join(images_dir, cls)
            if not os.path.exists(cls_path):
                raise Exception(f"❌ Missing class folder: {cls}")

            if len(os.listdir(cls_path)) == 0:
                raise Exception(f"❌ No images in {cls}")

            print(f"✅ {cls} verified")

        labels = np.load(labels_path, allow_pickle=True)
        print(f"✅ Labels loaded: {len(labels)} samples")

        print("🎉 Data Validation Successful")
