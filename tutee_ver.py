# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:08:06 2022

@author: ksh76
"""

import urllib3
import requests
import io
import urllib.request
from requests import get
import sys
import os
import numpy as np
import torch
from soft_dtw_cuda import SoftDTW
import json


def download(url, file_name_):
    # open in binary mode
    file_name = file_name_ + 'tutee' + '.mp4'
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        response.raise_for_status()

        # write to file
        file.write(response.content)
    
    return file_name    


def VIBE(file_name):    
    os.system("python demo.py --vid_file " + file_name + " --output_folder output/  --no_render")


def softdtw(tutor_url): #넘파이 파일
    tutor_r = requests.get(tutor_url) # 넘파이 파일임   
    
    tutor_r.raise_for_status()  
    
    tutor = np.load(io.BytesIO(tutor_r.content))
    #튜티 넘파이 불러오기 
    tutee = np.load('./save_coordin.npy') 
    
   #frame 조절 kp 25
    frame = 0
    if (tutor.shape[0] < tutee.shape[0]):
        frame = tutor.shape[0]
                   
        tutor = tutor[:,:25,:]
        tutee = tutee[:frame,:25,:]
        
    tutor_th = torch.from_numpy(tutor).cuda()
    tutee_th = torch.from_numpy(tutee).cuda()
    
    #compare
    sdtw = SoftDTW(use_cuda=True, gamma=0.1)
    dance_distance = sdtw(tutor_th,tutee_th)/frame
    return frame, dance_distance

# 가공 여기서    
def gotoback(frame, dance_distance):
    WIN_SIZE = 3 # 0.125초 피드백?
    sz = int(frame/WIN_SIZE)   
    flag = 0     
    print(sz)
    sz_md = frame % WIN_SIZE
    print(sz_md)
    
    if(sz_md > 0):
        gotoback = torch.zeros(sz+1)
        flag = np.zeros(sz+1)
        for i in range(sz+1):
            if(i <= sz):
              gotoback[i] = torch.mean(dance_distance[WIN_SIZE*i : WIN_SIZE*i+WIN_SIZE], axis = 0)
            else:
              gotoback[i] = torch.mean(dance_distance[i : ], axis = 0)
    else:
        gotoback = torch.zeros(sz)
        flag = np.zeros(sz)
        for i in range(sz):
            if(i <= sz):
              gotoback[i] = torch.mean(dance_distance[WIN_SIZE*i : WIN_SIZE*i+WIN_SIZE], axis = 0)
   
    
    print(gotoback.shape)
    
    #상중하        
    for j in range(len(gotoback)):
        if(gotoback[j] < 100):
            flag[j] = 0
        elif(gotoback[j] >= 100 and gotoback[j] < 300):
            flag[j] = 1
        elif(gotoback[j] >= 300 and gotoback[j] < 700):
            flag[j] = 2
        else:
            flag[j] = 3
    
  
    flag = np.int64(flag)
    print(flag)
    count = 24
    eva = int(sz/count)
    eva_mod = sz % count


    if(eva_mod > 0): 
        gotoback2 = np.zeros(eva+1)
        gotoback2 = np.int64(gotoback2)
        for k in range(eva+1):
            if(k <= eva):
              gotoback2[k] = np.argmax(np.histogram(flag[count*k : count*k+count], bins = range(5))[0])
            else:
              gotoback2[k] =  np.argmax(np.histogram(flag[k : ], bins = range(5))[0])
    else:
        gotoback2 = np.zeros(eva)
        gotoback2 = np.int64(gotoback2)
        for k in range(eva):
            if(k <= eva):
              gotoback2[k] =  np.argmax(np.histogram(flag[count*k : count*k+count], bins = range(5))[0])
    
    print(gotoback2)    
        
    
    #serialization
    lists = gotoback2.tolist()
    # json_str = json.dumps(lists)
    return lists
    
    
if __name__ == '__main__':
    file_name = download(sys.argv[3],sys.argv[2]) #tutor npy url/tutee id/tutee s3 url
    VIBE(file_name)
    
    #softdtw
    frame, dist = softdtw(sys.argv[1])
    lists = gotoback(frame, dist)
    print(lists)
    
    # TO DO:
    # 백으로 json_str, tutee id 전달

    headers = {'Content-Type': 'application/json; charset=utf-8', }

    datas = {'feedback_result': lists, 'tutee_id': sys.argv[2]}
    response = requests.post('http://52.79.67.201:8000/deep/tutee', data = json.dumps(datas), headers = headers)

    print(response.status_code)
    
