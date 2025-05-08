import cv2
from ultralytics import YOLO
import numpy as np

def func_box_bh(video_name: str, save_box_name: str, start_frame: int, end_frame: int):

    model = YOLO("./data/weights/ball_handler.pt")
    bboxes = []
    batch_size = 1

    i_frame = 0

    video_capture = cv2.VideoCapture()
    if video_capture.open(video_name):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        length = end_frame - start_frame

    while i_frame < length:
        imgs = []
        for i in range(batch_size):
            ret, frame = video_capture.read()
            if not ret:
                break
            imgs.append(frame)
        results = model(imgs, stream=True, classes=[0], conf=0.5, iou=0.8)
        for result in results:
            ball_handler_index = None
            boxes = result.boxes.cpu().numpy()
            for index, conf in enumerate(boxes.conf):
                if boxes.cls[index] == 0:
                    bboxes.append(np.hstack([start_frame + i_frame, boxes.xyxy[index], conf]))

            i_frame += 1
    print(bboxes)
    np.save(save_box_name, bboxes, 0, 1)

    
# func_box_bh("./../../basket_short.mp4", "./data/annotations/test.npy", 18, 19)
func_box_bh("./../../basket_short.mp4", "./data/annotations/ball_handler.npy", 0, 2000)
