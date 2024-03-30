import os
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
from ultralytics.solutions import distance_calculation
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

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

def track():
    # Load the YOLO model
    model = YOLO('yolov8n.pt')

    # Initialize video capture from webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    while True:
        _, img = cap.read()
        
        # BGR to RGB conversion is performed under the hood
        # see: https://github.com/ultralytics/ultralytics/issues/2575
        results = model.predict(img)

        for r in results:
            annotator = Annotator(img)
            
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, model.names[int(c)], color='red')  # Change color to red
          
        img = annotator.result()  
        cv2.imshow('Object Detection', img)     

        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()
#     # results = model.predict(source=0, show=True, tracker="bytetrack.yaml")

if __name__ == "__main__":
    #preTrain()
    #train()
    #results()
    track()
