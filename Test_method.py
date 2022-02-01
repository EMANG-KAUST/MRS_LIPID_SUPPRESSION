#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 21:04:34 2022

@author: Maria de los Angeles Gomez
"""


from BaselineRemoval import BaselineRemoval

import numpy as np
from numpy import trapz
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras

ppm_path="./ppm.xlsx"
model = keras.models.load_model("./BiLSTM_results/BiLSTMa.h5")

def read_file(path):
    file=pd.read_excel(path,header=None)
    data=file.values
    DATA=[]

    for d in data:
        baseObj=BaselineRemoval(d)
        data_p=baseObj.ZhangFit()
        DATA.append(data_p)

    DATABASE=np.array(DATA)

    DATA_N=DATABASE-np.mean(DATABASE)/np.std(DATABASE)
    DATA_N=np.expand_dims(DATA_N,2)
    return DATA_N

def window(Data,ppm):
    idx = np.where(ppm < 4)[0]
    ppm1=ppm[idx]
    idx2 = np.where(ppm1 > -2)[0]
    ppm2=ppm[idx2]
    data=Data[:,idx2]
    return data, ppm2
#------------------------------SCSA Test-----------------------------
MRS_path_test_MRSC = "./noiseTest/MRS_C.xlsx"
MRS_path_test_MRS = "./noiseTest/MRS.xlsx"
MRS_path_test_SCSA = "./noiseTest/SCSA.xlsx"

mrsc=read_file(MRS_path_test_MRSC)
mrs=read_file(MRS_path_test_MRS)
mrs_scsa=read_file(MRS_path_test_SCSA)

file=pd.read_excel(ppm_path,header=None)
ppm=file.values
ppm=np.array(ppm)

mrsc_w, ppm2=window(mrsc,ppm)
mrs_w, ppm2=window(mrs,ppm)
mrs_scsa_w, ppm2=window(mrs_scsa,ppm)

SCSA_pred=model.predict(mrs_scsa_w)
MRS_pred=model.predict(mrs_w)

idx = np.where(ppm2 < 1.5)[0]
PPM1=ppm2[idx]
idx2 = np.where(PPM1 > 0.9)[0]
PPM=ppm2[idx2]

sir_lac_s=[]
sir_lac_m=[]

DIF_s_a=[]
DIF_m_a=[]
DIF_s_sir=[]
DIF_m_sir=[]
SNR=[5, 10, 15, 20, 50]
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in np.arange(0, 5):
  # Get the sample and the reconstruction
  pure = np.array(mrsc_w[1])
  reconstruction_s = np.array(SCSA_pred[i])
  reconstruction_m = np.array(MRS_pred[i])
  pure_a = trapz(pure[idx2], axis=0)
  reconstruction_s_a =trapz(reconstruction_s[idx2],axis=0) 
  reconstruction_m_a =trapz(reconstruction_m[idx2],axis=0)
  diff_s=np.abs((pure_a-reconstruction_s_a)/pure_a)
  diff_m=np.abs((pure_a-reconstruction_m_a)/pure_a)
  
  print("---------SNR="+str(SNR[i])+"-----------")
  print("Amplitude GT: "+str(pure_a))
  print("Amplitude BiLSTM without SCSA:  "+str(reconstruction_m_a))
  print("Amplitude BiLSTM with SCSA:  "+str(reconstruction_s_a))
  print("AR without SCSA: "+str(diff_m))
  print("AR with SCSA: "+str(diff_s))
  DIF_s_a.append(diff_s)
  DIF_m_a.append(diff_m)
  
  pure_lac1=pure[idx2][10]
  pure_lac2=pure[idx2][13]
  print("Signal Intensity GT Lac1: "+str(pure_lac1))
  print("Signal Intensity GT Lac2: "+str(pure_lac2))
  reconstruction_m_lac1=reconstruction_m[idx2][10]
  reconstruction_m_lac2=reconstruction_m[idx2][13]
  print("Signal Intensity BiLSTM without SCSA Lac1: "+str(reconstruction_m_lac1))
  print("Signal Intensity BiLSTM without SCSA Lac2: "+str(reconstruction_m_lac2))
  reconstruction_s_lac1=reconstruction_s[idx2][10]
  reconstruction_s_lac2=reconstruction_s[idx2][13]
  print("Signal Intensity BiLSTM with SCSA Lac1: "+str(reconstruction_s_lac1))
  print("Signal Intensity BiLSTM with SCSA Lac2: "+str(reconstruction_s_lac2))
  
  sir_s_lac1=abs((pure_lac1-reconstruction_s_lac1)/pure_lac1)
  sir_m_lac1=abs((pure_lac1-reconstruction_m_lac1)/pure_lac1)
  print("SIR without SCSA Lac1 "+str(sir_m_lac1))
  print("SIR with SCSA Lac1 "+str(sir_s_lac1))
  sir_lac_s.append(sir_s_lac1)
  sir_lac_m.append(sir_m_lac1)
  sir_s_lac2=abs((pure_lac2-reconstruction_s_lac2)/pure_lac2)
  sir_m_lac2=abs((pure_lac2-reconstruction_m_lac2)/pure_lac2)
  print("SIR without SCSA Lac2 "+str(sir_m_lac2))
  print("SIR with SCSA Lac2 "+str(sir_s_lac2))
  sir_lac_s.append(sir_s_lac2)
  sir_lac_m.append(sir_s_lac2)

  
  # Plot GT and reconstruciton with and without SCSA
  axes[i].plot(ppm2,pure,label="Ground truth")
  axes[i].plot(ppm2,reconstruction_s,color='red',label="BiLSTM with SCSA")
  axes[i].plot(ppm2,reconstruction_m,color='orange',label="BiLSTM without SCSA")
  axes[i].set_title('SNR='+str(SNR[i]))
  axes[i].set_xlabel('ppm')
  axes[i].legend()
  
differece_s_a=np.mean(DIF_s_a)
differece_m_a=np.mean(DIF_m_a)
dif_std_s_a=np.std(DIF_s_a)
dif_std_m_a=np.std(DIF_m_a)

print("Mean Difference AR without SCSA: "+str(differece_m_a)+"+/-"+str(dif_std_m_a))
print("Mean Difference AR with SCSA: "+str(differece_s_a)+"+/-"+str(dif_std_s_a))

differece_lac_s=np.mean(sir_lac_s)
dif_std_lac_s=np.std(sir_lac_s)
differece_lac_m=np.mean(sir_lac_m)
dif_std_lac_m=np.std(sir_lac_m)

print("Mean Difference SIR without SCSA: "+str(differece_lac_m)+"+/-"+str(dif_std_lac_m))
print("Mean Difference SIR with SCSA: "+str(differece_lac_s)+"+/-"+str(dif_std_lac_s))

#-----------------------------------Amplitude Test-----------------------------------
MRS_path_amp_test_MRSC = "./amplitudTest/MRS_C.xlsx"
MRS_path_amp_test_MRS = "./amplitudTest/MRS.xlsx"
MRS_path_amp_test_SCSA = "./amplitudTest/SCSA.xlsx"

mrsc=read_file(MRS_path_amp_test_MRSC)
mrs=read_file(MRS_path_amp_test_MRS)
mrs_scsa=read_file(MRS_path_amp_test_SCSA)

file=pd.read_excel(ppm_path,header=None)
ppm=file.values
ppm=np.array(ppm)

mrsc_w, ppm2=window(mrsc,ppm)
mrs_w, ppm2=window(mrs,ppm)
mrs_scsa_w, ppm2=window(mrs_scsa,ppm)

SCSA_pred=model.predict(mrs_scsa_w)
MRS_pred=model.predict(mrs_w)

idx = np.where(ppm2 < 1.5)[0]
PPM1=ppm2[idx]
idx2 = np.where(PPM1 > 0.9)[0]
PPM=ppm2[idx2]

sir_lac_s=[]
sir_lac_m=[]

DIF_s_a=[]
DIF_m_a=[]
DIF_s_sir=[]
DIF_m_sir=[]
a=[2, 3, 5]
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in np.arange(0, 3):
  # Get the sample and the reconstruction
  pure = np.array(mrsc_w[1])
  reconstruction_s = np.array(SCSA_pred[i])
  reconstruction_m = np.array(MRS_pred[i])
  
  print("---------alpha="+str(a[i])+"-----------")
  
  pure_lac1=pure[idx2][10]
  pure_lac2=pure[idx2][13]
  print("Signal Intensity GT Lac1: "+str(pure_lac1))
  print("Signal Intensity GT Lac2: "+str(pure_lac2))
  reconstruction_s_lac1=reconstruction_s[idx2][10]
  reconstruction_s_lac2=reconstruction_s[idx2][13]
  print("Signal Intensity BiLSTM Lac1: "+str(reconstruction_s_lac1))
  print("Signal Intensity BiLSTM Lac2: "+str(reconstruction_s_lac2))
  
  sir_s_lac1=abs((pure_lac1-reconstruction_s_lac1)/pure_lac1)
  print("SIR Lac1 "+str(sir_s_lac1))
  sir_lac_s.append(sir_s_lac1)
  sir_s_lac2=abs((pure_lac2-reconstruction_s_lac2)/pure_lac2)
  print("SIR Lac2 "+str(sir_s_lac2))
  sir_lac_s.append(sir_s_lac2)
  
  # Plot GT and reconstruciton with and without SCSA
  axes[i].plot(ppm2,pure,label="Ground truth")
  axes[i].plot(ppm2,reconstruction_s,color='red',label="BiLSTM with SCSA")
  axes[i].set_title('alpha='+str(a[i]))
  axes[i].set_xlabel('ppm')
  axes[i].legend()
  
differece_s_a=np.mean(DIF_s_a)
dif_std_s_a=np.std(DIF_s_a)

differece_lac_s=np.mean(sir_lac_s)
dif_std_lac_s=np.std(sir_lac_s)

print("Mean Difference SIR: "+str(differece_lac_s)+" +/- "+str(dif_std_lac_s))


#-----------------------------------NOISE TEST-----------------------------------
MRS_path_amp_test_MRSC = "./noiseTest/MRS_C.xlsx"
MRS_path_amp_test_MRS = "./noiseTest/MRS.xlsx"
MRS_path_amp_test_SCSA = "./noiseTest/SCSA.xlsx"

mrsc=read_file(MRS_path_amp_test_MRSC)
mrs=read_file(MRS_path_amp_test_MRS)
mrs_scsa=read_file(MRS_path_amp_test_SCSA)

file=pd.read_excel(ppm_path,header=None)
ppm=file.values
ppm=np.array(ppm)

mrsc_w, ppm2=window(mrsc,ppm)
mrs_w, ppm2=window(mrs,ppm)
mrs_scsa_w, ppm2=window(mrs_scsa,ppm)

SCSA_pred=model.predict(mrs_scsa_w)
MRS_pred=model.predict(mrs_w)

idx = np.where(ppm2 < 1.5)[0]
PPM1=ppm2[idx]
idx2 = np.where(PPM1 > 0.9)[0]
PPM=ppm2[idx2]

sir_lac_s=[]
sir_lac_m=[]

DIF_s_a=[]
DIF_m_a=[]
DIF_s_sir=[]
DIF_m_sir=[]
SNR=[5, 10, 15]
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in np.arange(0, 3):
  # Get the sample and the reconstruction
  pure = np.array(mrsc_w[1])
  reconstruction_s = np.array(SCSA_pred[i])
  reconstruction_m = np.array(MRS_pred[i])
  
  print("---------SNR="+str(SNR[i])+"-----------")
  
  pure_lac1=pure[idx2][10]
  pure_lac2=pure[idx2][13]
  print("Signal Intensity GT Lac1: "+str(pure_lac1))
  print("Signal Intensity GT Lac2: "+str(pure_lac2))
  reconstruction_s_lac1=reconstruction_s[idx2][10]
  reconstruction_s_lac2=reconstruction_s[idx2][13]
  print("Signal Intensity BiLSTM Lac1: "+str(reconstruction_s_lac1))
  print("Signal Intensity BiLSTM Lac2: "+str(reconstruction_s_lac2))
  
  sir_s_lac1=abs((pure_lac1-reconstruction_s_lac1)/pure_lac1)
  print("SIR Lac1 "+str(sir_s_lac1))
  sir_lac_s.append(sir_s_lac1)
  sir_s_lac2=abs((pure_lac2-reconstruction_s_lac2)/pure_lac2)
  print("SIR Lac2 "+str(sir_s_lac2))
  sir_lac_s.append(sir_s_lac2)
  
  # Plot GT and reconstruciton with and without SCSA
  axes[i].plot(ppm2,pure,label="Ground truth")
  axes[i].plot(ppm2,reconstruction_s,color='red',label="BiLSTM with SCSA")
  axes[i].set_title('alpha='+str(a[i]))
  axes[i].set_xlabel('ppm')
  axes[i].legend()
  
differece_s_a=np.mean(DIF_s_a)
dif_std_s_a=np.std(DIF_s_a)

differece_lac_s=np.mean(sir_lac_s)
dif_std_lac_s=np.std(sir_lac_s)

print("Mean Difference SIR: "+str(differece_lac_s)+" +/- "+str(dif_std_lac_s))
