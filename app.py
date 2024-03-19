from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, \
    Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.utils import CustomObjectScope
from keras import backend as K
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from rembg import remove
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
app = Flask(__name__)


def dice_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (2.0 * intersection + 1.0) / (union + 1.0)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return intersection / (union + K.epsilon())


dic = {0: 'Non-Anemic', 1: 'Anemic'}
# Load model with custom loss and metric functions
model = load_model('model.h5', custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef, 'iou': iou})
modelAnemia = load_model('model_fold_3.h5')
model.make_predict_function()
modelAnemia.make_predict_function()


def predict_mask(imgPath):
    i = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    i = cv2.resize(i, (256, 256))

    # Normalize pixel values
    i = i / 255.0

    # Convert data type to float32
    i = i.astype(np.float32)

    # Add batch dimension
    i = np.expand_dims(i, axis=0)

    # Now, you can use the processed image with your model
    p = model.predict(i)[0] > 0.5
    p = p.astype(np.int32)
    # p = np.concatenate([p, p, p], axis=-1)
    p = p * 255
    return p


def segment_image(x, y):
    x = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    y = cv2.imread(y)
    y = cv2.resize(y, (256, 256))
    y = y.astype(np.int32)

    for i in range(len(x) - 1):
        for j in range(len(x) - 1):
            for k in range(3):
                if x[i][j][k] != 0:
                    x[i][j][k] = y[i][j][k]
    #x = cv2.resize(x, (64, 64))
    return x


def crop_image(imgPath):
    img = cv2.imread(imgPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the top-left and bottom-right points of the bounding box
    top_left = (x, y)
    bottom_right = (x + w, y + h)

    # Crop the ROI from the original image
    cropped_image = img[y:y + h, x:x + w]
    return cropped_image


def predict_anemia(imgPath):
    img = cv2.imread(imgPath)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    p = modelAnemia.predict(img)
    per = p[0][0]
    p = np.round(p).astype(int)[0][0]
    if p == 1:
        per = per*100
    else:
        per = (1-per)*100
    return dic[p], '{:.2f}'.format(per)


@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("index.html")


@app.route("/process_form", methods=['POST'])
def get_image():
    pred = None
    pred1 = None
    con = None
    img_path = None
    anemia = None
    if request.method == 'POST':
        try:
            i = random.randint(1, 1000)
            img = request.files['inputImage']
            img_path = "static/" + img.filename
            img.save(img_path)
            p = predict_mask(img_path)
            pred = "static/prediction/" + str(i) + ".png"
            pred1 = "static/pred/" + str(i) + ".png"
            con = "static/conj/" + str(i) + ".png"
            p1 = segment_image(p, img_path)
            cv2.imwrite(pred, p)
            cv2.imwrite(pred1, p1)
            crop = crop_image(pred1)
            cv2.imwrite(con, crop)
            anemia = predict_anemia(con)
        except KeyError:
            print("No 'inputImage' file in the request.")
    return render_template("index.html", img1=img_path, img2=pred, img3=con, prediction=anemia[0], percentage=anemia[1])


if __name__ == '__main__':
    app.run(debug=True)
