from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time
import pdb


def get_bags_of_sifts(image_paths,step):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
                                                                    
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    
    # You will want to construct SIFT features here in the same way you        #
    # did in build_vocabulary.m (except for possibly changing the sampling     #
    # rate) and then assign each local feature to its nearest cluster center   #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################    
     with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    len_img = len(image_paths)
    image_feats = np.zeros((len_img, vocab_size))
    for idx, path in enumerate(image_paths):
        img = np.asarray(Image.open(path) , dtype='float32')
        frames, descriptors = dsift(img, step = step, fast=True)
        d = distance.cdist(vocab, descriptors, 'euclidean')
        nn_dist = np.argmin(d, axis=0)
        h, bins = np.histogram(nn_dist, bins=range(0,vocab_size+1))
        norm = np.linalg.norm(h, ord=1)
        if norm==0:
            image_feats[idx,:] = h
        else:
            image_feats[idx,:] = h/norm
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats
