import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

def preTrain():
    # Downloading the weights    
    model = YOLO("yolov8n.pt") 
    
def train():
    names = ['drone']
    M = list(range(len(names)))
    class_map = dict(zip(M, names))
    
    # Define the command as a string
    model = YOLO("yolov8n.pt") 
    model.train(data="data.yaml", epochs=20, imgsz=640, workers=8, batch=16,save_period=10)
    
def results():
    paths2 = []
    for dirname, _, filenames in os.walk('./runs/detect/train'):
        for filename in filenames:
            if filename[-4:]=='.jpg':
                paths2+=[(os.path.join(dirname, filename))]
    paths2=sorted(paths2)
    
    for path in paths2:
        image = Image.open(path)
        image=np.array(image)
        plt.figure(figsize=(20,10))
        plt.imshow(image)
        plt.show()
    
def predict():
    best_path='runs/detect/train5/weights/best.pt'
    source='../datasets/test'
    save_dir = 'predicted_images'
    
    model = YOLO(best_path)
    results = model.predict(source=source, show=True, save_dir=save_dir) # Display preds. Accepts all YOLO predict arguments
    
def track():
    best_path='runs/detect/train4/weights/best.pt'
    video = "test/test7.mp4"
    #best_path='yolov8n.pt'
    model = YOLO(best_path)
    results = model.predict(source=0, show=True, tracker="bytetrack.yaml")
    
    
if __name__ == "__main__":
    #preTrain()
    #train()
    #results()
    #predict()
    track()
    
    
