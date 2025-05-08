from ultralytics import YOLO

# Load a model
model = YOLO("pie-tactic/yolo12l_waugments_500_box_loss/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
