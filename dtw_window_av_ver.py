# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 14:56:57 2022


"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:04:48 2022


"""
import os
cur_dir = os.getcwd()
import numpy as np
import torch
from soft_dtw_cuda import SoftDTW

tutor = np.load('./danmer_cv-main/save_tutor_coordin.npy')
tutee = np.load('./danmer_cv-main/save_tutee_coordin.npy')[:627]
dun = np.load('./danmer_cv-main/save_dundun_coordin.npy')[:627]
# hm3 = np.load('./danmer_cv-main/save_heymama3_coordin.npy')[:627]
# hm4 = np.load('./danmer_cv-main/save_heymama4_coordin.npy')[:627]


threshold = 100
WIN_SIZE = 5

tutor_ma = np.zeros(shape=(threshold,49,2))
tutee_ma = np.zeros(shape=(threshold,49,2))
# hm3_ma = np.zeros(shape=(threshold,49,2))
# dun_ma = np.zeros(shape=(threshold,49,2))
# hm4_ma = np.zeros(shape=(threshold,49,2))
# ddd = np.zeros(shape=(threshold,49,2)) 
    
for i in range(len(tutor)):
  #print(i)
  if i >= threshold:
   print("Limit")
   break
#print(np.mean(tutor[WIN_SIZE*i : WIN_SIZE*i+WIN_SIZE], axis=0))
  tutor_ma[i] = np.mean(tutor[WIN_SIZE*i : WIN_SIZE*i+WIN_SIZE], axis=0)
#print(tutor_ma)

for i in range(len(tutee)):
  #print(i)
  if i >= threshold:
   print("Limit")
   break
  #print(np.mean(tutee[WIN_SIZE*i : WIN_SIZE*i+WIN_SIZE], axis=0))
  tutee_ma[i] = np.mean(tutee[WIN_SIZE*i : WIN_SIZE*i+WIN_SIZE], axis=0)
#   #print(tutee_ma)


tutor_th = torch.from_numpy(tutor_ma).cuda()
tutee_th = torch.from_numpy(tutee_ma).cuda()
dun_th = torch.from_numpy(dun).cuda()

# hm3_th = torch.from_numpy(hm3).cuda()
# hm4_th = torch.from_numpy(hm4).cuda()

print(tutor_th.shape)

# # assert tutor_th.size(0) == tutee_th.size(0), "Match the length!"
sdtw = SoftDTW(use_cuda=True, gamma=0.1)
dance_distance = sdtw(tutor_th,tutee_th) / threshold
sim = dance_distance.mean()
print(sim)



# # dance_distance = sdtw(tutor_th,dun_th) / 627
# # print(dance_distance.mean())

# # # sdtw = SoftDTW(use_cuda=True, gamma=0.1)
# # # dance_distance = sdtw(tutor_th,hm3_th) / 627
# # # print(dance_distance.mean())

# # # sdtw = SoftDTW(use_cuda=True, gamma=0.1)
# # # dance_distance = sdtw(tutor_th,hm4_th) / 627
# # # print(dance_distance.mean())
