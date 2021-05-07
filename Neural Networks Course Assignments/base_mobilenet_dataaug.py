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

train1 = pd.read_csv(home_dir+'labels/train3.txt',header=None)
val1 = pd.read_csv(home_dir+'labels/validate3.txt',header=None)
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
BATCH_SIZE = 128    # Choose a batch size
NUM_CLASSES = 23  # For MINC-2500 Dataset, 23 classes

datagen = ImageDataGenerator(  
   # Data Augmentation Methods Go Here
   preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input, # Mutually exclusive with rescale
   horizontal_flip=True,
   vertical_flip=True,
   rotation_range=30,
   brightness_range=[0.7,1.3],
   zoom_range=0.3
)  

# Retrieve MINC training data 
train_gen = datagen.flow_from_dataframe(  
    train1,
    x_col='x',
    y_col='class',
    directory=None,
    target_size=(imgsize,imgsize),   # Resize to 224
    batch_size= BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42,   
)

valid_gen = datagen.flow_from_dataframe(
    val1,
    x_col='x',
    y_col='class',
    directory=None,
    target_size=(imgsize,imgsize),
    batch_size= BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
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

# Training the MobileNet on MINC
EPOCHS = 18   # Train in batches of 18 epochs since SCSE max 4hrs
VALID_STEPS = valid_gen.n//valid_gen.batch_size
TRAIN_STEPS = train_gen.n//train_gen.batch_size

def lr_scheduler(epoch, lr):
    decay_rate = 0.25
    decay_step = 5
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

cb_checkpointer = ModelCheckpoint(filepath = '/home/UG/user/mobilenet_aug.hdf5', 
                                  monitor = 'val_loss',
                                  save_best_only = True, save_weights_only = True, mode = 'auto')  # To save weights for reuse

cb_LRscheduler = LearningRateScheduler(lr_scheduler, verbose=1) # Reduce LR when performance plateaus

model.load_weights('/home/UG/user/mobilenet_aug.hdf5') 

history = model.fit(train_gen,
                        epochs = EPOCHS, 
                        verbose = 2,
                        validation_data = valid_gen,
                        validation_steps = VALID_STEPS,
                        steps_per_epoch = TRAIN_STEPS,
                        callbacks = [cb_checkpointer, cb_LRscheduler]) 
                        