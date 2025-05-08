from roboflow import Roboflow
from ultralytics import YOLO

# Downloading the dataset from roboflowl
rf = Roboflow(api_key="lKtSaPoqALQa1knCbvkD")
project = rf.workspace( "piebasket").project("ball_handler_insep")
version = project.version(2)
dataset = version.download("yolov12")

# Initializing model
model = YOLO('yolo12l.yaml')

# Defining search space
search_space = {
            "lr0": (1e-5, 1e-1),
            "lrf": (0.01, 1.0),
            "momentum": (0.6, 0.98),
            "weight_decay": (0.0, 0.001),
            "box": (0.02, 0.2)
        }

# Searches the best hyperparameters
model.tune(data=f"{dataset.location}/data.yaml", batch=0.7, epochs=10, iterations=300, space=search_space, plots=False, save=False, val=False)
