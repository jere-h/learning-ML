import numpy as np # linear algebra
import warnings
warnings.filterwarnings('ignore') # filter warnings
import os
import cv2
from keras.applications import imagenet_utils
from keras.applications import MobileNet
from keras.utils import np_utils
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Import Training Data (manual capture, MobileNet model retraining)
dataset = []
IMG_DATA_PATH = 'downloads' # Folder of training images
img_size = 224
# Using datagen.flow instead of flow from directory here
for directory in os.listdir(IMG_DATA_PATH):
     path = os.path.join(IMG_DATA_PATH, directory)
     if not os.path.isdir(path):
         continue
     for item in os.listdir(path):
         if item.startswith("."): # to hopefully ignore hidden files
             continue
         img = cv2.imread(os.path.join(path, item))
         img = cv2.resize(img, (img_size, img_size)) # resizing image
#         img = image.img_to_array(img)
#         img_array_expanded_dims = np.expand_dims(img_array, axis=0)
         dataset.append([img, directory])
# dataset = [
#     [[...], 'label1'],
#     [[...], 'label2'],
#     ... ]

CLASS_MAP = {'pen':0, 'pencil':1}
num_classes = len(CLASS_MAP)

def maplabel(mapdata):
    return CLASS_MAP[mapdata]

data, labels = zip(*dataset)
labels = list(map(maplabel, labels)) # labels are now 1 2 3 4
labels = np_utils.to_categorical(labels) # 1hot encode labels

# Datagen flow takes care of standardization

# Split into Train-Test sets 
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(np.array(data),
                                                  np.array(labels),
                                                  test_size = 0.1, random_state=2)


# Visualize proportion in each class
# plt.figure(figsize=(15,7))
# g = sns.countplot(Y_train, palette="icefire")
# plt.title("Number of digit classes")
# Y_train.value_counts()

# plot some samples
# img = X_train.iloc[0].as_matrix()
# img = img.reshape((28,28))
# plt.imshow(img,cmap='gray')
# plt.title(train.iloc[0,0])
# plt.axis("off")
# plt.show()

# Data augmentation only on training set not val set
datagen_aug = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # dimesion reduction
         rotation_range=0.1,  # randomly rotate images in the range 10 degrees
         zoom_range = 0.1, # Randomly zoom image 10%
         width_shift_range=0.1,  # randomly shift images horizontally 10%
         height_shift_range=0.1,  # randomly shift images vertically 10%
         horizontal_flip=True  # randomly flip images
#         vertical_flip=False  # randomly flip images
         )
datagen_no_aug = ImageDataGenerator(
         width_shift_range=0.1,  # randomly shift images horizontally 10%
         height_shift_range=0.1  # randomly shift images vertically 10%
         )

train_gen = datagen_aug.flow(
    X_train, 
    Y_train, 
    batch_size = 24)

valid_gen = datagen_no_aug.flow(
        X_val,
        Y_val,
        batch_size = 24)

# Define the model eg Mobilenet base, add layers

model = Sequential()
model.add(MobileNet(input_shape=(224,224,3), include_top=False)) #layer 0
model.add(GlobalAveragePooling2D())
model.add(Dense(1024,activation='relu')) #add dense layers
model.add(Dense(1024,activation='relu')) #dense layer 2
model.add(Dense(512,activation='relu'))#dense layer 3
model.add(Dense(2,activation='softmax'))
# model.layers[0].trainable = False

# Compile model with optimizer
model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Training
epochs = 5
batch_size = 50

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = 3)
cb_checkpointer = ModelCheckpoint(filepath = 'cnntest2checkpt.hdf5', 
                                  monitor = 'val_loss',
                                  save_best_only = True, mode = 'auto')

history = model.fit_generator(train_gen,
                              epochs = epochs, 
                              validation_data = valid_gen,
                              validation_steps = 10,
                              steps_per_epoch = train_gen.n//train_gen.batch_size, # depends on amount of training data
                              callbacks=[cb_checkpointer, cb_early_stopper])
model.load_weights('cnntest2checkpt.hdf5')
# Save model for later use
model.save("cnntest2.h5")