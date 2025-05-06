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
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        results = model(imgs)
        for result in results:
            ball_handler_index = None
            boxes = result.boxes.numpy()
            for index, conf in enumerate(boxes.conf):
                if boxes.cls[index] == 0:
                    if ball_handler_index is None:
                        ball_handler_index = index
                    else:
                        if conf > boxes.conf[ball_handler_index]:
                            ball_handler_index = index
            if ball_handler_index is not None:
                bboxes.append(np.hstack([i_frame, boxes.xyxy[ball_handler_index], boxes.conf[ball_handler_index]]))

            i_frame += 1
    print(bboxes)
    np.save(save_box_name, bboxes, 0, 1)


# func_box_bh("./../../basket_short.mp4", "./data/annotations/test.npy", 18, 19)
func_box_bh("./../../basket_short.mp4", "./test.npy", 10, 20)
