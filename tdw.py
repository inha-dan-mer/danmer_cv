# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 21:56:35 2022

@author: ksh76
"""


import numpy as np
from numpy import dot
from numpy.linalg import norm

from scipy.spatial.distance import euclidean
import dtw
from fastdtw import fastdtw

tutor = np.load('C:/Users/ksh76/capstone/save_tutor_coordin.npy')
tutee = np.load('C:/Users/ksh76/capstone/save_tutee_coordin.npy')
dun = np.load('C:/Users/ksh76/capstone/save_dundun_coordin.npy')

if __name__ == "__main__":
    threshold = 100

    tutor_ma = np.zeros(shape=(threshold,49*2))
    tutee_ma = np.zeros(shape=(threshold,49*2))
    
    tutor_ma = tutor
    tutee_ma = tutee
        
    # tutor_ma = np.reshape(tutor, (len(tutor),49*2)) #627*98
    # tutee_ma = np.reshape(tutee, (len(tutee),49*2))
    # dun_ma = np.reshape(dun, (len(dun),49*2))   
    
    
    dd = dtw.dtw(tutor_ma,tutee_ma, keep_internals=True).distance
   

    # distance, path = fastdtw(tutor_ma, tutee_ma, dist=euclidean)
    # print(distance)
    