import os
import cv2
from ultralytics import YOLO

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
model = YOLO("../runs/detect/train/weights/best.pt")

save_path = "../results/"
video_path = "../test_video.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

os.makedirs(save_path, exist_ok=True)
output_path = os.path.join(save_path, 'output.mp4')
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

accident_counter = 0
consecutive_threshold = 5
confidence_threshold = 0.5
show_accident = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, verbose=False, conf=0.5)
    result = results[0]

    has_accident = False
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        conf = box.conf[0].item()

        if class_id == "Accident" and conf > confidence_threshold:
            has_accident = True
            break

    if has_accident:
        accident_counter += 1
    else:
        accident_counter = 0

    if accident_counter >= consecutive_threshold:
        show_accident = True

    if show_accident:
        annotated_frame = result.plot()

        cv2.imshow("Accident Detection", annotated_frame)
        out.write(annotated_frame)

        if not has_accident:
            show_accident = False
    else:
        cv2.imshow("Accident Detection", frame)
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved to {output_path}")