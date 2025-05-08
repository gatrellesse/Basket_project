import comet_ml
from roboflow import Roboflow
from ultralytics import YOLO

# Downloading the dataset from roboflow
rf = Roboflow(api_key="lKtSaPoqALQa1knCbvkD")
project = rf.workspace( "piebasket").project("ball_handler_insep")
version = project.version(3)
dataset = version.download("yolov12")

# Logging experiment to comet
comet_ml.init()
exp = comet_ml.Experiment(api_key="IfsXBfZcpIhkSlITbnQaPcvKK", workspace="gelmi", project_name="pie-tactic") 

# Initializing model
model = YOLO('yolo12m.yaml')

# Training model
results = model.train(data=f'{dataset.location}/data.yaml', epochs=500, patience=50, batch=0.7, box=1.0, cls=3.0, device=0, project="pie-tactic", name="yolo12l_waugments_500_box_loss")
