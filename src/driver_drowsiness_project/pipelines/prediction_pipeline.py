import cv2
import numpy as np
from tensorflow.keras.models import load_model



class DrowsinessPredictor:
    def __init__(self,model_path,img_size=224):
        self.model = load_model(model_path)
        self.img_size = img_size
        self.class_name = ["Open","Closed","yawn","no_yawn"]
    def preprocess(self,frame):
        img = cv2.resize(frame,(self.img_size,self.img_size))
        img = img / 255.0
        img = np.expand_dims(img,axis=0)
        return img
    
    def predict(self,frame):
        img = self.preprocess(frame)
        preds = self.model.predict(img,verbose=0)
        print("Pred is :",preds)
        class_id = np.argmax(preds)
        confidence = preds[0][class_id]

        return self.class_name[class_id],float(confidence)


        

