# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 21:36:46 2017

@author: HGY
"""

import numpy as np
from scipy.io import loadmat


#%% SIFTSimpleMatcher function
def SIFTSimpleMatcher(descriptor1, descriptor2, THRESH=0.7):
    '''
    SIFTSimpleMatcher 
    Match one set of SIFT descriptors (descriptor1) to another set of
    descriptors (decriptor2).
    Each descriptor from descriptor1 can at
    most be matched to one member of descriptor2, but descriptors from
    descriptor2 can be matched more than once.
    
    Matches are determined as follows:
    For each descriptor vector in descriptor1, find the Euclidean distance
    between it and each descriptor vector in descriptor2.
    
    If the smallest distance is less than thresh*(the next smallest distance), we say that
    the two vectors are a match, and we add the row [d1 index, d2 index] to
    the "match" array.
    
    INPUT:
    - descriptor1: N1 * 128 matrix, each row is a SIFT descriptor.1128*128
    - descriptor2: N2 * 128 matrix, each row is a SIFT descriptor.613*128
    - thresh: a given threshold of ratio. Typically 0.7
    
    OUTPUT:
    - Match: N * 2 matrix, each row is a match. For example, Match[k, :] = [i, j] means i-th descriptor in
        descriptor1 is matched to j-th descriptor in descriptor2.
    '''

    #############################################################################
    #                                                                           #
    #                              YOUR CODE HERE                               #
    #                                                                           #
    #############################################################################
    listmatch = [] 
    for i in range(len(descriptor1)):#1128
        temp=[]
        for j in range(len(descriptor2)):#613
            distance = np.linalg.norm(descriptor1[i] - descriptor2[j]) #Euclidean distance descriptor1[i] - descriptor2[j]128
            temp.append(distance) 
   
        index = np.argsort(temp) #Sorted by the index value of the array
        if temp[index[0]] < THRESH * temp[index[1]]:#if [index 0] value is less than tresh times [index 1]
            listmatch.append([i, index[0]])#The append () method is used to add a new object to the end of the list.
            
    match = np.array(listmatch)#21*2 

    
    
    #############################################################################
    #                                                                           #
    #                             END OF YOUR CODE                              #
    #                                                                           #
    #############################################################################   
    
    return match
