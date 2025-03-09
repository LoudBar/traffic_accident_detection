import os

from ultralytics import YOLO

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

model = YOLO("../base.pt")
model.to("cuda")

model.train(
    data='data.yaml',
    epochs=100,
    patience=20,
    batch=16,
    imgsz=640,
    device=0,
    workers=8,
    single_cls=True,
    pretrained=True,
    optimizer='auto',
    cos_lr=True,
    lr0=0.01,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    mosaic=1.0,
    mixup=0.1,
    augment=True,
    cache=False,
    val=True,
    save=True,
    save_period=10,
    freeze=22
)
