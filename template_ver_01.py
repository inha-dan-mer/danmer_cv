# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 01:05:23 2022

@author: ksh76
"""

import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance


#tutor tutee
def cosine_similarity(list_1, list_2):
  cos_sim = dot(list_1, list_2) / (norm(list_1) * norm(list_2))
  return cos_sim



tutor = np.load('C:/Users/ksh76/capstone/save_tutor_coordin.npy')
tutee = np.load('C:/Users/ksh76/capstone/save_tutee_coordin.npy')
dun = np.load('C:/Users/ksh76/capstone/save_dundun_coordin.npy')

if __name__ == "__main__":

    # Example
    threshold = 100
    key_no = 49
    # tutor_ma = np.zeros(shape=(threshold,49*2))
    # tutee_ma = np.zeros(shape=(threshold,49*2))
    # dun_ma = np.zeros(shape=(threshold,49,2))
    
    tutor = np.reshape(tutor, (len(tutor),49*2)) #627*98
    tutee = np.reshape(tutee, (len(tutee),49*2))
    dun = np.reshape(dun, (len(dun),49*2))   
 
               
        
    # 2450 - 2450: tutor - tutee

    similarity = 0.
   
    for j in range(threshold):
        #similarity += cosine_similarity(tutor[j], dun[j])
       #similarity += distance.euclidean(tutor[j],dun[j]) 
        similarity += distance.euclidean(tutor[j],tutee[j]) 
    
    similarity = similarity / threshold
    
 
        
        
        
        
    