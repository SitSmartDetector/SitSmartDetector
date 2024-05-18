from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from PIL import Image
import numpy as np
import joblib

# Load the pretrained model
model = YOLO('yolov8x.pt')

# Fine-tune the model on the mixed dataset with a lower learning rate
# results = model.train(data='custom.yaml', epochs=50, lr0=0.001, freeze=10)

sample_path = r"sample.png"

results = model.predict(
    source=sample_path,
    mode="predict",
    device="cpu",
    classes=[0, 56, 62, 63], # using result.names or model.names
    conf=0.6,
    # save_txt = True,
    #stream=True,
    # save=True,
    project="output_test",
    name="without"
    # Name=['person', 'chair', 'tv', 'laptop']
)

# Print the prediction results
box = results[0].boxes

x0, y0 = None, None
x56, y56, w56, h56 = None, None, None, None

for b in box:
    cls = int(b.cls.item())
    if cls == 0:
        xy = b.xywh.tolist()[0][:2]  # 提取 xy 部分
        x0, y0 = xy[0], xy[1]
    elif cls == 56:
        xywh = b.xywh.tolist()[0]  # 提取 xywh
        x56, y56, w56, h56 = xywh[0], xywh[1], xywh[2], xywh[3]

if x0 and y0 and x56 and y56 and w56 and h56:
    distance = np.sqrt(((x0 - x56)/x56)**2 + ((y0 - y56)/y56)**2)
    normalized_distance = distance/(w56)
else:
    distance = -1 # one or both of the class is missing
    normalized_distance = -1

model = joblib.load('logistic_regression_model.pkl')
normalized_distance_reshaped = np.array(normalized_distance).reshape(1, -1)
y_pred = model.predict(normalized_distance_reshaped)


print("x0, y0:", x0, y0)
print("x56, y56, w56, h56:", x56, y56, w56, h56)
print(normalized_distance)
print(y_pred[0])

# for r in results:
#     boxes = r.boxes
#     conf = boxes.conf
#     print(conf)
#     for box in boxes:
#         b = box.xywh#[0]  # get box coordinates in (left, top, right, bottom) format
#         conf = box.conf
#         cID = box.cls
#         print("bounding box=", b, "confidence=", conf, "class=", cID, "className=", model.names[int(cID)])

print('Finish!')