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



WIN_SIZE = 5

tutor = np.load('C:/Users/ksh76/capstone/save_tutor_coordin.npy')
tutee = np.load('C:/Users/ksh76/capstone/save_tutee_coordin.npy')
dun = np.load('C:/Users/ksh76/capstone/save_dundun_coordin.npy')

if __name__ == "__main__":

    # Example
    threshold = 100
    key_no = 49

    
    tutor_ma = np.zeros(shape=(threshold,49,2))
    tutee_ma = np.zeros(shape=(threshold,49,2))
    dun_ma = np.zeros(shape=(threshold,49,2))
    
    for i in range(len(tutor)):
        print(i)
        if i >= threshold:
            print("Limit")
            break
        #print(np.mean(tutor[WIN_SIZE*i : WIN_SIZE*i+WIN_SIZE], axis=0))
        tutor_ma[i] = np.mean(tutor[WIN_SIZE*i : WIN_SIZE*i+WIN_SIZE], axis=0)
        print(tutor_ma)
    
    for i in range(len(tutee)):
      print(i)
      if i >= threshold:
        print("Limit")
        break
      #print(np.mean(tutee[WIN_SIZE*i : WIN_SIZE*i+WIN_SIZE], axis=0))
      tutee_ma[i] = np.mean(tutee[WIN_SIZE*i : WIN_SIZE*i+WIN_SIZE], axis=0)
      print(tutee_ma) 
      
    for i in range(len(dun)):
      print(i)
      if i >= threshold:
        print("Limit")
        break
      #print(np.mean(tutee[WIN_SIZE*i : WIN_SIZE*i+WIN_SIZE], axis=0))
      dun_ma[i] = np.mean(dun[WIN_SIZE*i : WIN_SIZE*i+WIN_SIZE], axis=0)
      print(dun_ma) 

               
        
    # 2450 - 2450: tutor - tutee
    similarity = 0.
    for i in range(threshold):
        for j in range(49):
            #similarity += cosine_similarity(tutor_ma[i,j], tutee_ma[i,j])
            similarity += distance.euclidean(tutor_ma[i,j], dun_ma[i,j])          
    
    
    similarity = similarity / (threshold*49)   
 
    
    # ---
    #TODO():
    # 1) 전 처리 과정에서 convolve 활용
    # 2) Similarity 방식 고도화. (49,2)를 기준으로 유사도 계산? 
    # -> Similairty with global perspective
    # 2번까지 진행한 뒤, ablation study 수행하기
    # +) Neuran network (train network를 통해 유사도를 출력)
    # 향후에 NN이 필요하게될 가능성 언급 (최종보고서 때)
    # ---
        
        
        
        
    