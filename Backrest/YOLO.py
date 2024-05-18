from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from PIL import Image

# Load the pretrained model
model = YOLO('YOLOv8x.pt')

# Fine-tune the model on the mixed dataset with a lower learning rate
# results = model.train(data='custom.yaml', epochs=50, lr0=0.001, freeze=10)

sample_path = r"IMG_4732.png"

results = model.predict(
    source=sample_path,
    mode="predict",
    device="cpu",
    classes=[0, 56, 62, 63], # using result.names or model.names
    conf=0.6,
    # save_txt = True,
    stream=True,
    save=True,
    project="output_test",
    name="without"
    # Name=['person', 'chair', 'tv', 'laptop']
)

# Print the prediction results
for r in results:
    boxes = r.boxes
    conf = boxes.conf
    for box in boxes:
        b = box.xywh#[0]  # get box coordinates in (left, top, right, bottom) format
        conf = box.conf
        cID = box.cls
        print("bounding box=", b, "confidence=", conf, "class=", cID, "className=", model.names[int(cID)])

print('Finish!')