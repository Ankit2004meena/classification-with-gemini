import os
import numpy as np
from ultralytics import YOLO
from PIL import Image

model_path = os.path.join('model', 'best.pt')
print('MODEL PATH:', model_path)
print('EXISTS:', os.path.exists(model_path))
model = YOLO(model_path)
print('MODEL NAMES:', model.names)
print('NUM CLASSES:', len(model.names))

img = Image.new('RGB', (224, 224), 'white')
arr = np.array(img, dtype=np.uint8)
results = model.predict(source=arr, verbose=False)
print('RESULTS TYPE:', type(results))
print('RESULTS LEN:', len(results))
print('RESULT0 TYPE:', type(results[0]))
print('HAS PROBS:', hasattr(results[0], 'probs'))
print('PROBS TYPE:', type(results[0].probs))
print('PROBS REPR:', repr(results[0].probs)[:1000])
try:
    arr = np.array(results[0].probs, dtype=np.float32)
    print('NP ARRAY SHAPE:', arr.shape)
    print('NP ARRAY DTYPE:', arr.dtype)
    print('NP ARRAY:', arr)
except Exception as e:
    print('NP ARRAY ERROR:', type(e).__name__, e)
