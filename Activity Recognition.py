# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 14:47:08 2021

@author: ahmed
"""

import numpy as np
from numpy import dstack
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from tqdm import tqdm
import os
import gc
import random

from numba import jit, cuda
# import code
# code.interact(local=locals)


import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc

from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D
from keras.layers import LSTM, TimeDistributed, Dropout, Flatten, Dense
from tensorflow.keras.layers import ConvLSTM2D
from keras import backend as K

from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.wrappers.scikit_learn import KerasClassifier

import tensorflow as tf
from tensorflow.python.client import device_lib


print(device_lib.list_local_devices())



features = pd.read_csv("E:/Software/professional practice projects/In progress 4/UCI HAR Dataset/features.txt", delim_whitespace=True, header=None)

print(features.head())

activity_labels = pd.read_csv("E:/Software/professional practice projects/In progress 4/UCI HAR Dataset/activity_labels.txt", header=None, delim_whitespace=True, index_col=0)

print(activity_labels)



def load_file(path):
    file = pd.read_csv(path, header=None, delim_whitespace=True)
    return file.values


def load_dataset(path):
    files = list()
    for file_name in tqdm(os.listdir(path)):
        file_path = os.path.join(path, file_name)
        file = load_file(file_path)
        files.append(file)
    
    # stacking them so that the features are the 3d dimension
    files = dstack(files)
    
    return files


x_train_path = "E:/Software/professional practice projects/In progress 4/UCI HAR Dataset/train/Inertial Signals"
x_test_path  = "E:/Software/professional practice projects/In progress 4/UCI HAR Dataset/test/Inertial Signals"

y_train_path = "E:/Software/professional practice projects/In progress 4/UCI HAR Dataset/train/y_train.txt"
y_test_path  = "E:/Software/professional practice projects/In progress 4/UCI HAR Dataset/test/y_test.txt"

x_train = load_dataset(x_train_path)
x_test = load_dataset(x_test_path)

y_train = load_file(y_train_path)
y_test = load_file(y_test_path)

print(np.min(y_train), ' --> ', np.max(y_train))

# zero-offset class values
y_train -= 1
y_test -= 1

print(np.min(y_train), ' --> ', np.max(y_train))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)


def grid_search(classifier, x_train):
    
    batch_size = [128, 512]
    epochs     = [30]
    validation_split = [0.1, 0.2]
    
    param_distributions = dict(batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    
    search = RandomizedSearchCV(estimator=classifier, param_distributions=param_distributions, n_jobs=1, verbose=2, cv=2)
    
    results = search.fit(x_train, y_train)
    
    print("best score: {} ==>  for params: {}".format(results.best_score_, results.best_params_))
    
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    
    for mean, std, param in zip(means, stds, params):
        print("Mean: {},  Std: {}, Params: {}".format(means, stds, params))
    
    return results


cnt = 0

def train_model(classifier, x_train, y_train, grid_params):
    
    batch_size, epochs = grid_params['batch_size'], grid_params['epochs']
    validation_split = grid_params['validation_split']
    
    global cnt
    cnt += 1
    my_callbacks = [ModelCheckpoint(filepath='model_'+str(cnt), monitor='val_loss', save_best_only=True, verbose=1), 
                    EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
                    CSVLogger('model_' + str(cnt) + '.csv')]
    
    model = classifier()
    
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_split=validation_split, callbacks=my_callbacks)
    
    return model, history.history


def evaluate_model(model, x_test, y_test, history):
    
    loss, accuracy = model.evaluate(x_test, y_test)
    
    plt.figure(figsize=(8, 4))
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Activity Recognition Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train_accuracy', 'val_accurcay'], loc='lower right')
    plt.show()
    
    plt.figure(figsize=(8, 4))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Activity Recognition Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train_Loss', 'val_Loss'], loc='upper right')
    plt.show()
    
    return loss, accuracy




# @cuda.jit(target ="cuda")
def run(classifier, x_train, x_test):
   
    model = KerasClassifier(classifier, verbose=1)
    results = grid_search(model, x_train)
    
    trained_model, history = train_model(classifier, x_train, y_train, results.best_params_)
    loss, accuracy = evaluate_model(trained_model, x_test, y_test, history)
    
    print('accuracy = {}'.format(round(accuracy * 100.0, 2)))
    
    return trained_model



n_timesteps = x_train.shape[1]
n_features = x_train.shape[2]
n_outputs = y_train.shape[1]


initial_learning_rate = 0.00001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)


"""   LSTM Model  """

def model_1():
    
    K.clear_session()
    
    model_input = Input(shape=(n_timesteps, n_features))
    
    x = LSTM(512)(model_input)
    x = Dropout(0.25)(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.15)(x)
    
    output = Dense(n_outputs, activation='softmax')(x)
    
    model = Model(inputs=model_input, outputs=output)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    


model = run(model_1, x_train, x_test)

model.save('LSTM model.h5')


n_steps, n_length = 4, 32
trainX = x_train.reshape((x_train.shape[0], n_steps, n_length, n_features))
testX = x_test.reshape((x_test.shape[0], n_steps, n_length, n_features))

def model_2():
    
    K.clear_session()
    
    model_input = Input(shape=(None, n_length, n_features))
    
    x = TimeDistributed(Conv1D(64, kernel_size=3, activation='relu'))(model_input)
    x = TimeDistributed(Conv1D(64, kernel_size=3, activation='relu'))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(Flatten())(x)
    
    x = LSTM(100)(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu')(x)
    
    output = Dense(n_outputs, activation='softmax')(x)
    
    model = Model(inputs=model_input, outputs=output)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


model2 = run(model_2, trainX, testX)

model2.save('CNN-LSTM model.h5')

# model2.summary()

# gc.collect()



# Confusion Matrix

# def confusion_matrix(model, X_test, activities):
    
#     y_pred=model.predict(X_test)
#     y_predict=pd.Series([activities[i+1] for i in np.argmax(y_pred, axis=1)])
#     y_actual=pd.Series([activities[i+1] for i in np.argmax(y_test, axis=1)])
    
    
#     # Code for drawing seaborn heatmaps
#     class_names = list(activities.values())
#     df_heatmap = pd.DataFrame(confusion_matrix(y_actual, y_predict) )
    
#     plt.style.use('seaborn-whitegrid')
#     sns.set(style="white", font_scale = 1.5)
    
#     fig = plt.figure(figsize=(15,12))
#     heatmap = sns.heatmap(df_heatmap, annot=True, fmt="d", cmap="Blues")
    
#     # Setting tick labels for heatmap
#     heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
#     heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right')
#     heatmap.set_xticklabels(class_names)
#     heatmap.set_yticklabels(class_names)
#     plt.ylabel('Actual Activity')
#     plt.xlabel('Predicted Activity')
#     plt.title("Confusion Matrix\n")
    
#     plt.show()



import keras

model2 = keras.models.load_model('CNN-LSTM model.h5')

model2.summary()

# confusion_matrix(model2, testX, activity_labels)



activities = {1: 'Walking',    
             2: 'Walking Upstairs',
             3: 'Walking Downstairs',
             4: 'Sitting',
             5: 'Standing',
             6: 'Laying'}


def plot_confusion_matrix(model, testX, activities):
    
    y_pred=model.predict(testX)
    # y_predict=pd.Series([activities[i+1] for i in np.argmax(y_pred, axis=1)])
    # y_actual=pd.Series([activities[i+1] for i in np.argmax(y_test, axis=1)])
    
    y_predict=[activities[i+1] for i in np.argmax(y_pred, axis=1)]
    y_actual=[activities[i+1] for i in np.argmax(y_test, axis=1)]
    
    
    # Code for drawing seaborn heatmaps
    class_names = list(activities.values())
    df_heatmap = confusion_matrix(y_actual, y_predict)
    
    plt.style.use('seaborn-whitegrid')
    sns.set(style="white", font_scale = 1.5)
    
    fig = plt.figure(figsize=(15, 12))
    heatmap = sns.heatmap(df_heatmap, annot=True, fmt="d", cmap="Blues")
    
    # Setting tick labels for heatmap
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    heatmap.set_xticklabels(class_names)
    heatmap.set_yticklabels(class_names)
    plt.ylabel('Actual Activity')
    plt.xlabel('Predicted Activity')
    plt.title("Confusion Matrix\n")
    
    plt.show()
    
    return y_pred, df_heatmap


y_pred, df_heatmap = plot_confusion_matrix(model2, testX, activities)

# ROC AUC

def plot_roc(y_pred):
    n_classes = 6

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    #Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.style.use('seaborn-whitegrid')
    sns.set(style="white", font_scale = 1.2)
    plt.figure(figsize=(12,12))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = ['aqua', 'darkorange', 'cornflowerblue', "red", "green", "black"]
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='{0} (area = {1:0.2f})'
                 ''.format(activities[i+1], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for each activity')
    plt.legend(loc="lower right")
    
    plt.show()


plot_roc(y_pred)



