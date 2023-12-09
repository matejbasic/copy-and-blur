import os

models_dir_path = f"{os.path.dirname(__file__)}/models"
detector_weights_url = "https://drive.google.com/uc?id=1CGNCkZQNj5WkP3rLpENWAOgrBQkUWRdw"
detector_weights = f"{models_dir_path}/yolov8x_person_face.pt"
checkpoint_url = "https://drive.google.com/uc?id=11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4"
checkpoint = f"{models_dir_path}/mivolo_imbd.pth.tar"
