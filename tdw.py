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
hm3 = np.load('C:/Users/ksh76/capstone/danmer_cv-main/save_heymama3_coordin.npy')
hm4 = np.load('C:/Users/ksh76/capstone/danmer_cv-main/save_heymama4_coordin.npy')

if __name__ == "__main__":
    threshold = 627

    tutor_ma = np.zeros(shape=(threshold,49,2))
    tutee_ma = np.zeros(shape=(threshold,49,2))
    hm3_ma = np.zeros(shape=(threshold,49,2))
    dun_ma = np.zeros(shape=(threshold,49,2))
    hm4_ma = np.zeros(shape=(threshold,49,2))
    ddd = np.zeros(shape=(threshold,49,2))
    
    tutor_ma = tutor
    tutee_ma = tutee
    dun_ma = dun
    hm3_ma = hm3
    hm4_ma = hm4    
 
    
    #ddd = dtw.dtw(tutor_ma[0],hm4_ma[0], keep_internals=True).distance / 627
    
    for i in range(len(tutor)):
        ddd = dtw.dtw(tutor_ma[i], hm4_ma[i], keep_internals=True).distance
        
    ddd = ddd/threshold