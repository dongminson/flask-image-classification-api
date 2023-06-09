import base64
import json
import uuid
import os
import numpy as np

from dotenv import load_dotenv
from flask import Flask, request, jsonify, url_for
from io import BytesIO
from PIL import Image, ImageFile
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
model = MobileNet(weights='imagenet', include_top=True)

@app.route('/', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return 'Error: no image found'
    image_file = request.files['file']

    if image_file :
        img = Image.open(BytesIO(image_file.read()))
        img.load()
        height_str = os.getenv('IMAGE_HEIGHT')
        width_str = os.getenv('IMAGE_WIDTH')
        height = int(height_str)
        width = int(width_str)
        img = img.resize((width, height), Image.ANTIALIAS)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        pred = model.predict(x)
        itemList = decode_predictions(pred, top=5)
        items = []
        for item in itemList[0]:
            items.append({'name': item[1], 'probability': float(item[2])})
        response = {'prediction': items}
        return json.dumps(response)
    else:
        return 'Error: filetype is not supported'
if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 9000, app)
    