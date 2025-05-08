from ultralytics import YOLO
import numpy as np
import os
import shutil

model = YOLO("yolov8m.pt")

folder_name = 'images'
i = 1
for filename in os.listdir(folder_name):
    results = model.predict(f'{folder_name}/{filename}')
    results[0] = results[0].cpu().numpy()
    classes = results[0].boxes.cls.astype(np.int64)
    boxes = results[0].boxes.xywhn
    if 0 in classes:
        file = open(f'DatasetPie/labels/{i:09d}.txt', "x")
        for cls, box in zip(classes, boxes):
            if cls == 0: 
                file.write(f'{int(cls)}  {box[0]} {box[1]} {box[2]} {box[3]}\n')
        file.close()
        shutil.move(f'{folder_name}/{filename}', f'DatasetPie/images/{i:09d}.jpg')
        i = i + 1
