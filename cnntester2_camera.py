from keras.preprocessing import image
from keras.models import load_model
import cv2
import keras
import numpy as np
from keras.utils.generic_utils import CustomObjectScope
# MobileNet uses several custom functions. You need to use a
# CustomObjectScope to load a saved model 
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    
    model = load_model('cnntest2_mobilenet.h5')
    
class_labels = ('pen',
                'pencil')
 
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle of consideration
    cv2.rectangle(frame, (148, 148), (402, 402), (255, 255, 255), 2)
    
    # extract the region of image within the user rectangle
    roi = frame[150:400, 150:400]
    img = cv2.resize(roi, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # predict the captured input image from camera
    pred = model.predict(img)
    pred_index = np.argmax(pred[0])

    # display prediction
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Pred:" + class_labels[pred_index],
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("frame", frame)
    
    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()