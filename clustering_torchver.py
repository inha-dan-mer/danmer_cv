# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance

import torch
from fast_pytorch_kmeans import KMeans


kp_count = 49
tutor = np.load('save_tutor_coordin.npy')
tutee = np.load('save_tutee_coordin.npy')[:627]
tutor_th = torch.from_numpy(tutor).cuda()
tutee_th = torch.from_numpy(tutee).cuda()


closest_tutor = torch.zeros(size=(627,20,2)).cuda()
for j in range(2):  # No. frames
    
    kmeans = KMeans(n_clusters=20, mode='euclidean', verbose=1)

    labels = kmeans.fit_predict(tutor_th[j])
    centers = kmeans.centroids

    for idx in range(20):
        print('idx: ', idx)
        labels = labels.cpu()
        arridx = np.where(labels == idx)  # shape: [21]
        print('arridx: ', arridx)
        unique, counts = np.unique(labels, return_counts=True)

        cnt, tmp_dist = -1, 0
        for c in range(len(arridx[0])):
            d = torch.norm(centers[idx] - tutor_th[j,arridx[0][c],:].squeeze(), p=2)  # L2 distance
            print("L2 distance: ", d)
            if c == 0:
                fin_dist = d
                cnt = cnt + 1
            else:
                if d <= tmp_dist:
                    fin_dist = d
                    cnt = cnt + 1
                else:
                    pass
            tmp_dist = fin_dist
        print(cnt)
        print("\n")
        try:
            closest_tutor[j,idx] = tutor_th[j,arridx[0][cnt],:]
        except IndexError:
            pass
                
                
                
