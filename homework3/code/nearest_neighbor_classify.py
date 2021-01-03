from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    ''' 
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    k = 10
    M = len(test_image_feats)
    train_labels = np.array(train_labels)
    
    dist = distance.cdist(train_image_feats, test_image_feats)
    idx = np.argsort(dist, axis = 0)   
    allLab = np.unique(train_labels)
    countLab = np.zeros((len(allLab),M))
    for i in range(M):
        for j in range(len(allLab)):
            kLab = train_labels[idx[0:k,i]]
            countLab[j,i] = sum(allLab[j] == kLab)
    label_idx = np.argmax(countLab,axis=0)
    test_predicts = allLab[label_idx.astype(int)]
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
