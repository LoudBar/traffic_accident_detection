import os
from ultralytics import YOLO

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

model = YOLO("best.pt")

for k, v in model.named_parameters():
    print(k)