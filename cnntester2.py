from keras.models import load_model
from keras.preprocessing import image
import keras
import cv2
import numpy as np
import sys
from keras.utils.generic_utils import CustomObjectScope
# MobileNet uses several custom functions. You need to use a
# CustomObjectScope to load a saved model 
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model = load_model('cnntest2_mobilenet.h5')
    
# filepath = sys.argv[1] # if needed to specify test directory

class_labels = ('pen',
                'pencil')

img_path = 'test/pencil/1.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
pred_index = np.argmax(pred[0])
    
print('Predicted:', class_labels[pred_index])
print('Confidence:', pred[0][pred_index])

