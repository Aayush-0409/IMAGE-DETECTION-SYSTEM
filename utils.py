# utils.py
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import ( # type: ignore
    MobileNetV2,
    preprocess_input,
)
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.imagenet_utils import decode_predictions # type: ignore


model = MobileNetV2(weights="imagenet")

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def predict_image(img_path):
    img = prepare_image(img_path)
    preds = model.predict(img, verbose=0)
    decoded = decode_predictions(preds, top=1)[0][0]
    return f"{decoded[1]} ({decoded[2]*100:.2f}%)"
