import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

home_dir = '/home/user/minc-2500/'   # Change based on your directory filepath

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

imgsize = 224  # For resizing the original images
BATCH_SIZE = 128 
NUM_CLASSES = 23  # For MINC-2500 Dataset

# Defining a training process train on MINC2500
datagen = ImageDataGenerator( 
    preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input, # Mutually exclusive with rescale
#    horizontal_flip=True,
#    vertical_flip=True,
#    rotation_range=20,
#    brightness_range=[0.7,1.3],
#    zoom_range=0.2
)  

# Retrieve MINC2500 training data from directory
train_gen = datagen.flow_from_dataframe(  
    train1,
    x_col='x',
    y_col='class',
    directory=None,
    validate_filenames=False,
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
    validate_filenames=False,
    target_size=(imgsize,imgsize),
    batch_size= BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
)

# Using mobilenetv2 as the model
base_model = tf.keras.applications.MobileNetV2(include_top = False, weights='imagenet', input_shape=(imgsize,imgsize,3))

for layer in base_model.layers[:37]:  # Freeze blocks 1 - 3 
  layer.trainable = False

inputs = tf.keras.Input(shape=(imgsize, imgsize, 3))
x = base_model(inputs)  # No training=false here because I want to train the other layers
x = tf.keras.layers.GlobalAveragePooling2D()(x) 
x = tf.keras.layers.Dropout(0.5)(x)                          
x = tf.keras.layers.Dense(128, kernel_initializer=tf.keras.initializers.GlorotUniform(),  # Xavier Uniform Initialization
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)  # weight decay as used in gluoncv example
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
premodel = tf.keras.Model(inputs,outputs)

# Compiling the model
premodel.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),  
              metrics=['accuracy'])

# Training the model 

EPOCHS = 25      # EPOCHS TRAINED SO FAR
VALID_STEPS = valid_gen.n//valid_gen.batch_size
TRAIN_STEPS = train_gen.n//train_gen.batch_size

cb_checkpointer = ModelCheckpoint(filepath = '/home/user/ensemble/mobilenet_trf.hdf5', 
                                  monitor = 'val_loss',
                                  save_best_only = True, 
                                  save_weights_only=True,
                                  mode = 'auto')  # Checkpoint callback to save best weights for reuse


cb_reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.25,
                              patience=3, min_lr=0.00001)   # Reduce LR when loss increases for 3 straight epochs

# premodel.load_weights('/home/user/ensemble/mobilenet_trf.hdf5')   # Train from loaded weights

history = premodel.fit(train_gen,
                        epochs = EPOCHS, 
                        validation_data = valid_gen,
                        validation_steps = VALID_STEPS,
                        steps_per_epoch = TRAIN_STEPS,
                        verbose=2,
                        callbacks=[cb_checkpointer, cb_reduceLR])   
