#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:55:24 2022

@author: Maria de los Angeles Gomez
"""

from BaselineRemoval import BaselineRemoval
import numpy as np
import pandas as pd

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