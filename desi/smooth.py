#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
train=np.load('/home/bwang/dataset/lowsnr-trainset.npy',allow_pickle=True).item()
ids=list(train.keys())
def smooth_flux(flux,kernal_size):
    flux_sm15=[]
    half=kernal_size//2
    for i in range(0,len(flux)):
        if i < half:
            flux_sm15.append(flux[i])
        #print(i)
        if i > len(flux)-half-1:
            flux_sm15.append(flux[i])
        #print(i)
        if (i>=half) & (i<=len(flux)-half-1):
        #print(i)
            flux_sm15.append(np.mean([flux[i-half:i+half]]))
    return flux_sm15

matrix=[]
labels_classifier=[]
labels_offset=[]
col_density=[]
for id in ids:
    flux_matrix=[]
    flux=train[id]['FLUX']
    labels_classifier.append(train[id]['labels_classifier'])
    labels_offset.append(train[id]['labels_offset'])
    col_density.append(train[id]['col_density'])
    for sample in flux:#sample是片段flux
        smooth3=smooth_flux(sample,kernal_size=3)
        smooth7=smooth_flux(sample,kernal_size=7)
        smooth15=smooth_flux(sample,kernal_size=15)
        flux_matrix.append(np.array([sample,smooth3,smooth7,smooth15]))
    matrix.append(flux_matrix)
dataset={ids[i]:{'FLUX_MATRIX':matrix[i],'labels_classifier':  labels_classifier[i], 'labels_offset':labels_offset[i] , 'col_density': col_density[i]} for i in range(0,len(ids))}
np.save('/home/bwang/dataset/%s-matrixtrainset.npy'%(len(ids)),dataset)

