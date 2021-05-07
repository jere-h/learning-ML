# Define the correct model, load the correct weights, set the correct file paths
# Set the correct data filepaths eg labels/train3.txt according to what you trained on
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

imgsize = 224  # 224 Mobilenet
NUM_CLASSES = 23  # For MINC-2500 Dataset, 23 classes

# Load the extracted features that you have extracted from the penultimate layer
train_features = np.load('/home/UG/user/aug_train_features.npy')   # Change the filepath based on the model
val_features = np.load('/home/UG/user/aug_val_features.npy')

save_dir = '/home/UG/user/'       # Change to your own save directory if needed

# Define the Model to use: eg MobileNetV2
#  -- Change This Part based on the model you are testing, also load the correct weights below
basemodel = tf.keras.applications.MobileNetV2(
    include_top=True,
    weights=None,
    classes=NUM_CLASSES,
)

# Compile 
basemodel.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),  
              metrics=['accuracy'])
basemodel.load_weights('/home/UG/user/mobilenet_aug.hdf5')  # Load the correct weights 
print('loaded weights')

# Define the SVM, Random Forest, & Softmax Classifiers -> Train them with the extracted features as inputs
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

labels = train1['class'].copy()
val_labels = val1['class'].copy()
le = LabelEncoder()
le.fit(labels)
val_labels = le.transform(val_labels)  

scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)   
# SVC Classifier
clf1 = SVC(probability=True, gamma=0.1, degree=4, decision_function_shape='ovo')
clf1.fit(train_features, labels)  # Fit the Support Vector Classifier 

joblib.dump(clf1, save_dir+'SVC_3.joblib')   # Save classifier 

print('predicting val_features')
output1 = clf1.predict_proba(val_features)
print('SVC acc: ', accuracy_score(val_labels, np.argmax(output1, axis=1)))

# Define RF Classifier
clf2 = RandomForestClassifier(criterion='gini')
 
clf2.fit(train_features, labels)

joblib.dump(clf2, save_dir+'RF_3.joblib')  # Save classifier 

output2 = clf2.predict_proba(val_features)
print('RF acc: ', accuracy_score(val_labels, np.argmax(output2, axis=1)))

# Softmax classifier in the paper refers to the original output of the initial model
inputsize = len(train_features[0])
inputs = tf.keras.layers.Input(shape=(inputsize,))
outputs = basemodel.layers[-1](inputs)   # Use pre-loaded fitted base model
model = tf.keras.models.Model(inputs,outputs)  # Softmax Classifier

output3 = model.predict(val_features, verbose=0)
print('Softmax acc: ', accuracy_score(val_labels, np.argmax(output3, axis=1)))

# Combine the 3 sets of outputs through Mean & Max
# Mean of the outputs (ie equal weightage)
mean_out = (output1 + output2 + output3)
mean_out = np.argmax(mean_out, axis=1)

# Alternatively, maximum of the outputs
max_out = np.fmax(output1, output2, output3)
max_out = np.argmax(max_out, axis=1)

print('Combined Mean Proba Acc Score: ', accuracy_score(val_labels, mean_out))  # For the mean
print('Max Proba Accuracy Score: ', accuracy_score(val_labels, max_out))  # For the max 

# Use differential evolution to search for optimal weighted average of outputs
from scipy.optimize import differential_evolution

def normalize(weights):
	# calculate l1 vector norm
	result = np.linalg.norm(weights, 1)
	# check for a vector of all zeros
	if result == 0.0:
		return weights
	# return normalized vector (unit norm)
	return weights / result

def ensemble_predictions(members, weights, testX):
	# make predictions
	yhats = [output1, output2, output3]
	yhats = np.array(yhats)
	# weighted sum across ensemble members
	summed = np.tensordot(yhats, weights, axes=((0),(0)))
	# argmax across classes
	result = np.argmax(summed, axis=1)
	return result
 
def evaluate_ensemble(members, weights, testX, testy):
	# make prediction
	yhat = ensemble_predictions(members, weights, testX)
	# calculate accuracy
	return accuracy_score(testy, yhat)
    
def loss_function(weights, members, testX, testy):
	# normalize weights
	normalized = normalize(weights)
	# calculate error rate
	return 1.0 - evaluate_ensemble(members, normalized, testX, testy)
    
 
labels = train1['class'].copy()
val_labels = val1['class'].copy()
le = LabelEncoder()
le.fit(labels)
val_labels = le.transform(val_labels)  
    

members = [clf1, clf2, model]
n_members = len(members)
# define bounds on each weight
bound_w = [(0.0, 1.0)  for _ in range(n_members)]
# arguments to the loss function
search_arg = (members, val_features, val_labels)
# global optimization of ensemble weights
result = differential_evolution(loss_function, bound_w, search_arg, maxiter=1000, tol=1e-7)
# get the chosen weights
weights = normalize(result['x'])
print('Optimized Weights: %s' % weights)
# evaluate chosen weights
score = evaluate_ensemble(members, weights, val_features, val_labels)
print('Optimized Weights Accuracy Score: %.3f' % score)
print('----')


# Per class precision, recall, F1 scores for the optimized weighted ensemble result
from sklearn.metrics import classification_report

print(classification_report(val_labels, ensemble_predictions(members, weights, val_features)))

class_list = ['brick', 'carpet', 'ceramic', 'fabric', 'foliage', 'food', 'glass', 'hair', 
              'leather', 'metal', 'mirror', 'other', 'painted', 'paper', 'plastic', 'polishedstone',
              'skin', 'sky', 'stone', 'tile', 'wallpaper', 'water', 'wood']