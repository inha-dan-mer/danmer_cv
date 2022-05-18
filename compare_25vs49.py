import os
import os
cur_dir = os.getcwd()
import numpy as np
import torch
from soft_dtw_cuda import SoftDTW

tutor = np.load('/content/drive/MyDrive/danmer_cv-main/save_tutor_coordin.npy')
tutee = np.load('/content/drive/MyDrive/danmer_cv-main/save_tutee_coordin.npy')[:627]
dun = np.load('/content/drive/MyDrive/danmer_cv-main/save_dundun_coordin.npy')[:627]
hm3 = np.load('/content/drive/MyDrive/danmer_cv-main/save_heymama3_coordin.npy')[:627]
hm4 = np.load('/content/drive/MyDrive/danmer_cv-main/save_heymama4_coordin.npy')[:627]
tutor2 = np.load('/content/drive/MyDrive/danmer_cv-main/save_tutor_coordin.npy')[:,:25,:]
tutee2 = np.load('/content/drive/MyDrive/danmer_cv-main/save_tutee_coordin.npy')[:627,:25,:]


tutor_th = torch.from_numpy(tutor).cuda()
tutee_th = torch.from_numpy(tutee).cuda()

tutor2_th = torch.from_numpy(tutor2).cuda()
tutee2_th = torch.from_numpy(tutee2).cuda()


dun_th = torch.from_numpy(dun).cuda()
# hm3_th = torch.from_numpy(hm3).cuda()
# hm4_th = torch.from_numpy(hm4).cuda()


# assert tutor_th.size(0) == tutee_th.size(0), "Match the length!"

import time
start = time.time()  # 시작 시간 저장
sdtw = SoftDTW(use_cuda=True, gamma=0.1)
dance_distance = sdtw(tutor_th,tutee_th)/627
print(dance_distance.mean())
print(time.time()-start)


print("-----------------------")

start = time.time()  # 시작 시간 저장
sdtw = SoftDTW(use_cuda=True, gamma=0.1)
dance_distance = sdtw(tutor2_th,tutee2_th)/627
print(dance_distance.mean())
print(time.time()-start)

# print(dance_distance)

# da = dance_distance.cpu().numpy()

# np.save('./dancedistance_tutortutee', da)

# print(da.shape)


# dance_distance = sdtw(tutor_th,dun_th) / 627
# print(dance_distance.mean())

# sdtw = SoftDTW(use_cuda=True, gamma=0.1)
# dance_distance = sdtw(tutor_th,hm3_th) / 627
# print(dance_distance.mean())

# sdtw = SoftDTW(use_cuda=True, gamma=0.1)
# dance_distance = sdtw(tutor_th,hm4_th) / 627
# print(dance_distance.mean())