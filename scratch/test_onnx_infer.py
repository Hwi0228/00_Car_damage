import cv2
import numpy as np
import onnxruntime as ort

# Load model
session = ort.InferenceSession('best.onnx')
input_name = session.get_inputs()[0].name

# Create dummy image or test image
img = np.zeros((320, 320, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), (250, 250, 250), -1)

# Preprocess
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(image_rgb, (320, 320))
input_tensor = resized.astype(np.float32) / 255.0
input_tensor = input_tensor.transpose(2, 0, 1)  # HWC to CHW
input_tensor = np.expand_dims(input_tensor, axis=0)  # BCHW

# Run inference
outputs = session.run(None, {input_name: input_tensor})
output0, output1 = outputs[0], outputs[1]

print("output0 shape:", output0.shape) # (1, 40, 2100)
print("output1 shape:", output1.shape) # (1, 32, 80, 80)

# Process detections
# output0 shape is (1, 40, 2100) -> transpose to (2100, 40)
preds = output0[0].T # (2100, 40)

boxes = []
scores = []
class_ids = []
mask_coeffs = []

num_classes = 4
class_names = ['Scratch', 'Dent', 'Breakage', 'Separated']

for i in range(preds.shape[0]):
    row = preds[i]
    cx, cy, w, h = row[0:4]
    class_scores = row[4:4+num_classes]
    max_class_id = np.argmax(class_scores)
    max_score = class_scores[max_class_id]
    
    if max_score > 0.25:
        x1 = cx - w / 2
        y1 = cy - h / 2
        boxes.append([x1, y1, w, h])
        scores.append(float(max_score))
        class_ids.append(int(max_class_id))
        mask_coeffs.append(row[4+num_classes:])

print(f"Found {len(boxes)} raw detections before NMS.")
