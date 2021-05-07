import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

home_dir = '/home/UG/user/minc-2500/'   # Change based on your directory filepath

train1 = pd.read_csv(home_dir+'labels/train3.txt',header=None)  # PUT CORRECT PATH
val1 = pd.read_csv(home_dir+'labels/validate3.txt',header=None) # I USED train3, check urs
test1 = pd.read_csv(home_dir+'labels/test3.txt',header=None)

train1['y'] = train1[0].str[7:10]        # Train-val-test in 85-5-10 split
val1['y'] = val1[0].str[7:10]
test1['y'] = test1[0].str[7:10]

train1.columns =['x', 'class'] 
val1.columns =['x', 'class'] 
test1.columns =['x', 'class'] 

train1.x = home_dir + train1.x
val1.x = home_dir + val1.x
test1.x = home_dir + test1.x

# x column has the filepaths, class column has the first 3 letters of each of the 23 classes

imgsize = 224  # 224 for MobileNet
NUM_CLASSES = 23  # For MINC-2500 Dataset, 23 classes

datagen = ImageDataGenerator(  
   preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input, # Mutually exclusive with rescale
   # NOTHING ELSE HERE, SINCE EXTRACTING FEATURES
)  

# Define the MobileNetV2
model = tf.keras.applications.MobileNetV2(
    include_top=True,
    weights=None,
    classes=NUM_CLASSES,
)

# Compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),  
              metrics=['accuracy'])

model.load_weights('/home/UG/xpan001/mobilenet_aug.hdf5')  # Load the correct weights

extract = Model(model.inputs, model.layers[-2].output)   # PENULTIMATE LAYER 

pred_gen_train = datagen.flow_from_dataframe(
    train1,
    x_col='x',
    directory=None,
    target_size=(imgsize,imgsize),
    batch_size= 5,
    class_mode=None,
    shuffle=False,
    validate_filenames=True,
)

pred_gen_val = datagen.flow_from_dataframe(
    val1,
    x_col='x',
    directory=None,
    target_size=(imgsize,imgsize),
    batch_size= 5,
    class_mode=None,
    shuffle=False,
    validate_filenames=True,
)

# Extract features from trained model
val_steps = pred_gen_val.n//pred_gen_val.batch_size
train_steps = pred_gen_train.n//pred_gen_train.batch_size

train_features = extract.predict(pred_gen_train, verbose=2, steps=train_steps)
val_features = extract.predict(pred_gen_val, verbose=2, steps=val_steps)

# Save the extracted features for later use   CHANGE FILEPATH
np.save('/home/UG/user/aug_train_features.npy', train_features)
np.save('/home/UG/user/aug_val_features.npy', val_features)