#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:13:54 2022

@author: Maria de los Angeles Gomez

This program makes the Bayesian search for the Hyperparameters Optimization  

inputs:
SCSA_path is the path of the file with the MRS with SCSA already applied
MRS_C_path is the path of the file with the ground truth of the MRS Spectras 
ppm_path is the path of the ppm file

output:
model is the best architecture found for the model already trained with 5000 
epochs and batch_size of 300. 

"""

import keras_tuner
from keras_tuner import BayesianOptimization
from BaselineRemoval import BaselineRemoval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
import keras.backend as K
from sklearn.model_selection import train_test_split
from functions_read_files import read_file, window

SCSA_path = "./DATA2000/SCSA.xlsx"
MRS_C_path = "./DATA2000/MRS_C.xlsx"
ppm_path="./DATA2000/ppm.xlsx"

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.reset_states()

def HP_BiLSTM(hp):
    def custom_loss_function(x, y):
        return  K.sqrt(K.sum(K.square(y - x), axis=-1))
    print('---Creating BiLSTM model---')
    num_conv = hp.Int('num_layers_conv', 1, 3)
    model = Sequential()
    for i in range(1, num_conv):
      model.add(Bidirectional(LSTM(units=hp.Int("units", min_value=32, max_value=256, step=32),return_sequences=True),input_shape=(203,1)))
      model.add(Dropout(hp.Float('dropout', 0, 0.5, step=0.1, default=0.5)))
      if hp.Boolean("BN"):
        model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(units=hp.Int("units", min_value=32, max_value=256, step=32),input_shape=(203,1))))
    model.add(Dropout(hp.Float('dropout', 0, 0.5, step=0.1, default=0.5)))
    if hp.Boolean("BN"):
        model.add(BatchNormalization())   
    model.add(Dense(203, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-5, 1e-2, sampling='log')),metrics=['mean_squared_error'])

    #print(model.summary())
    return model

x=read_file(SCSA_path)
y=read_file(MRS_C_path)

file=pd.read_excel(ppm_path)
ppm=file.values
ppm=np.array(ppm)

x_w, ppm2=window(x,ppm)
y_w, ppm2=window(y,ppm)
X_train, X_test, y_train, y_test = train_test_split(x_w, y_w, test_size=0.3, random_state=1)

tuner = BayesianOptimization(    
    HP_BiLSTM,
    objective=keras_tuner.Objective("val_loss", direction="min"),
    max_trials=10,
    num_initial_points=3,
    alpha=0.0001,
    beta=2.6,
    seed=42,
    directory="opt",
    project_name="BiLSTM", 
)

tuner.search_space_summary()
tuner.search(X_train, y_train, epochs=30, validation_split=0.3, callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3),CustomCallback()])
tuner.results_summary()
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=300, epochs=5000, callbacks=[CustomCallback()])
# "MSE"
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Generate reconstructions
num_reconstructions = len(X_test)
num=3
samples = X_test
pures = y_test
reconstructions = model.predict(samples)

idx = np.where(ppm < 4)[0]
ppm1=ppm[idx]
idx2 = np.where(ppm1 > -2)[0]

# Plot reconstructions
for i in np.arange(0, num):
  # Get the sample and the reconstruction
  sample = np.array(samples[i])
  pure = np.array(pures[i])
  reconstruction = np.array(reconstructions[i])
  fig, axes = plt.subplots(1, 3, figsize=(20, 5))
  # Plot sample and reconstruciton
  axes[0].plot(ppm[idx2],sample,label="Original")
  axes[0].set_title('MRS with lipid peak')
  axes[0].set_xlabel('ppm')
  axes[0].legend()
  axes[1].plot(ppm[idx2],reconstruction,color='red',label="BiLSTM output")
  axes[1].set_title('MRS without lipid peak')
  axes[1].set_xlabel('ppm')  
  axes[1].legend()
  axes[2].plot(ppm[idx2],pure,color='green',label="Ground truth")
  axes[2].set_title('MRS without lipid peak')
  axes[2].set_xlabel('ppm')  
  axes[2].legend()
  
model.evaluate(X_test, y_test)

