from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


#lokesh imports
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
#from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import os
import random
from keras import backend as K
#from sklearn.metrics import jaccard_similarity_score
#from sklearn.metrics import jaccard_score
#from shapely.geometry import MultiPolygon, Polygon
#import shapely.wkt
#import shapely.affinity
#from collections import defaultdict
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from keras import backend as keras
#import gc
#gc.collect()
#import warnings
#warnings.filterwarnings("ignore")
from tifffile import imread, imwrite
from skimage.transform import resize
import tensorflow as tf
import random as rn
import random

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'models/model_resnet.h5'
MODEL_PATH = 'models/unet.h5'
num_cls = 10
size = 160
smooth = 1e-12

def jaccard_coef(y_true, y_pred):
    """
    Jaccard Index: Intersection over Union.
    J(A,B) = |A∩B| / |A∪B| 
         = |A∩B| / |A|+|B|-|A∩B|
    """
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    total = K.sum(y_true + y_pred, axis=[0, -1, -2])
    union = total - intersection

    jac = (intersection + smooth) / (union+ smooth)

    return K.mean(jac)


#https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras



def get_unet():
    
    ##fixing numpy RS
    np.random.seed(42)

    ##fixing tensorflow RS
    #tf.random.set_seed(32)

    ##python RS
    rn.seed(12)
    


    inputs = Input((8, size, size))

    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 23),data_format='channels_first')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 43),data_format='channels_first')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first')(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 26),data_format='channels_first')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_uniform(seed= 45),data_format='channels_first')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first')(conv2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 54),data_format='channels_first')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 25),data_format='channels_first')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first')(conv3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 63),data_format='channels_first')(pool3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_uniform(seed= 32),data_format='channels_first')(conv4)
    drop4 = Dropout(0.5, seed= 38)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first')(drop4)

    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 32),data_format='channels_first')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 27),data_format='channels_first')(conv5)
    drop5 = Dropout(0.5, seed = 41)(conv5)

    up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 28),data_format='channels_first')(UpSampling2D(size = (2,2),data_format='channels_first')(drop5))
    merge6 = concatenate([drop4,up6], axis = 1)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 39),data_format='channels_first')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 21),data_format='channels_first')(conv6)

    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 11),data_format='channels_first')(UpSampling2D(size = (2,2),data_format='channels_first')(conv6))
    merge7 = concatenate([conv3,up7], axis = 1)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 17),data_format='channels_first')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 53),data_format='channels_first')(conv7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 63),data_format='channels_first')(UpSampling2D(size = (2,2),data_format='channels_first')(conv7))
    merge8 = concatenate([conv2,up8], axis = 1)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 29),data_format='channels_first')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 22),data_format='channels_first')(conv8)

    up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 54),data_format='channels_first')(UpSampling2D(size = (2,2),data_format='channels_first')(conv8))
    merge9 = concatenate([conv1,up9], axis = 1)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 67),data_format='channels_first')(merge9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 56),data_format='channels_first')(conv9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(seed= 64),data_format='channels_first')(conv9)
    conv10 = Conv2D(num_cls, (1, 1),strides=1, activation = 'sigmoid',data_format='channels_first')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer=Adam(lr=1e-4),loss='binary_crossentropy', metrics=[jaccard_coef])

    return model    

def adjust_contrast(bands, lower_percent=2, higher_percent=98):
    """
    to adjust the contrast of the image 
    bands is the image 
    """
    out = np.zeros_like(bands).astype(np.float32)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.float32)

def resize_image(image, model):
  """
  to resize the image
  """
  if image.shape == (837,837,8):
    return image

  else:
    resized_data = resize(image, (837,837,8))
    imwrite('resized.tif', resized_data, planarconfig='CONTIG')
    return tiff.imread("resized.tif")   

# Load your trained model
model = get_unet()
model.load_weights("models/unet_weights")
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    #rgb_img = image.load_img(img_path)

    ##########################################
    #https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
  
    #Read 16-band image
     
    rgb_img = tiff.imread(img_path)
    img = np.rollaxis(rgb_img, 0, 3)


    #resize the image according to model architecture
    img = resize_image(img, model)

    #adjust the contrast of the image
    x = adjust_contrast(img)
    
    cnv = np.zeros((960, 960, 8)).astype(np.float32)
    prd = np.zeros((num_cls, 960, 960)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x
     
    for i in range(0, 6):
        line = []
        for j in range(0, 6):
            line.append(cnv[i * size:(i + 1) * size, j * size:(j + 1) * size])
            
  
        x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
       # x = 2 * np.transpose(line, (0, 1, 2, 3)) - 1

        tmp = model.predict(x, batch_size=4)
       # tmp = np.transpose(tmp,(0,3,1,2))

        for j in range(tmp.shape[0]):
            prd[:, i * size:(i + 1) * size, j * size:(j + 1) * size] = tmp[j]
     
    # thresholds for each class 
    trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]
    for i in range(num_cls):
        prd[i] = prd[i] > trs[i]
    p = prd[:, :img.shape[0], :img.shape[1]] 
    print(p)  

    return p


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # # Process your result for human
        # # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        # return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

