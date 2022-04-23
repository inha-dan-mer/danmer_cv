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

tutor = np.load('C:/Users/ksh76/capstone/save_tutor_coordin.npy')
tutee = np.load('C:/Users/ksh76/capstone/save_tutee_coordin.npy')
dun = np.load('C:/Users/ksh76/capstone/save_dundun_coordin.npy')

if __name__ == "__main__":
    threshold = 100

    tutor_ma = np.zeros(shape=(threshold,49,2))
    tutee_ma = np.zeros(shape=(threshold,49,2))
    
    tutor_ma = tutor
    tutee_ma = tutee    
        
    # # tutor_ma = np.reshape(tutor, (len(tutor),49*2)) #627*98
    # # tutee_ma = np.reshape(tutee, (len(tutee),49*2))
    # # dun_ma = np.reshape(dun, (len(dun),49*2))      

    # # distance, path = fastdtw(tutor_ma, tutee_ma, dist=euclidean)
    # # print(distance)   
   
    # tutor_ma = np.reshape(tutor,2,-1)
    # tutee_ma = np.reshape(tutee, 2,-1)
    # dun_ma = np.reshape(dun, 2,-1)    
    
    #tutor_ma = np.reshape(tutor,(len(tutor),-1)
    
    ddd = dtw.dtw(tutor_ma,tutee_ma, keep_internals=True).distance
    for i in range(len(tutee)):
        dtw.dtw(tutor_ma[0], tutee_ma[0], keep_internals=True).distance
    # print(ddd)
    