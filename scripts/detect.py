import os

from ultralytics import YOLO

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
model = YOLO("../runs/detect/train/weights/best.pt")

save_path = "../results/"
video_path = "../test_video.mp4"

results = model.predict(source=video_path, project=save_path, save=True, show=True)