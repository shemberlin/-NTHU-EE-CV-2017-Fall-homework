import numpy as np
def my_imfilter(image, imfilter):

    '''
    Input:
        image: A 3d array represent the input image.
        imfilter: The gaussian filter.
    Output:
        output: The filtered image.
    '''
   
    ###################################################################################
    # TODO:                                                                           #
    # This function is intended to behave like the scipy.ndimage.filters.correlate    #
    # (2-D correlation is related to 2-D convolution by a 180 degree rotation         #
    # of the filter matrix.)                                                          #
    # Your function should work for color images. Simply filter each color            #
    # channel independently.                                                          #
    # Your function should work for filters of any width and height                   #
    # combination, as long as the width and height are odd (e.g. 1, 7, 9). This       #
    # restriction makes it unambigious which pixel in the filter is the center        #
    # pixel.                                                                          #
    # Boundary handling can be tricky. The filter can't be centered on pixels         #
    # at the image boundary without parts of the filter being out of bounds. You      #
    # should simply recreate the default behavior of scipy.signal.convolve2d --       #
    # pad the input image with zeros, and return a filtered image which matches the   #
    # input resolution. A better approach is to mirror the image content over the     #
    # boundaries for padding.                                                         #
    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can # 
    # see the desired behavior.                                                       #
    # When you write your actual solution, you can't use the convolution functions    #
    # from numpy scipy ... etc. (e.g. numpy.convolve, scipy.signal)                   #
    # Simply loop over all the pixels and do the actual computation.                  #
    # It might be slow.                                                               #
    ###################################################################################
    ###################################################################################
    # NOTE:                                                                           #
    # Some useful functions                                                           #
    #     numpy.pad or numpy.lib.pad                                                  #
    # #################################################################################
    
    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can 
    # see the desired behavior.

    # import scipy.ndimage as ndimage
    # output = np.zeros_like(image)
    # for ch in range(image.shape[2]):
    #     output[:,:,ch] = ndimage.filters.correlate(image[:,:,ch], imfilter, mode='constant')
    
    ###################################################################################
    ##                                 END OF YOUR CODE                              ##
    ###################################################################################
    #numpy.pad(array, pad_width, mode, **kwargs)[source]
    #Pads an array.

    #Parameters: 
    #array : array_like of rank N
    #Input array
    #pad_width : {sequence, array_like, int}
    #Number of values padded to the edges of each axis. ((before_1, after_1), ... (before_N, after_N)) unique pad widths for each axis. ((before, after),) yields same before and after pad for each axis. (pad,) or int is a shortcut for before = after = pad width for all axes.
    #mode : str or function
    #One of the following string values or a user supplied function.
    #‘constant’
    #Pads with a constant value.

    #Automatically fill 0 np.pad
    (width,height)=(int(image.shape[0]),int(image.shape[1]))#image size
    median=int((imfilter.shape[0]-1)/2)#filter mid
    udzeros=int(imfilter.shape[0]-1)#up down fill 0
    rgb=range(image.shape[2])#012
    filterm = (median, median)
    AzImg=np.zeros(((width+udzeros),(height+udzeros),3))
    for channel in rgb:        
        AzImg[:,:,channel] = np.pad(image[:,:,channel], pad_width=filterm, mode='constant', constant_values=0)

    #filter 180 degree rotation
    A = np.flip(imfilter,1)#Flip an array horizontally (axis=1).
    B = np.flip(A,0)#Flip an array vertically (axis=0).filter 180 degree rotation
    imfiltersize=imfilter.shape[0]

    output = np.zeros_like(image)

    for channel in rgb:#0~2
        for i in range(width):
            for j in range(height):
                output[i,j,channel] = np.sum(B * AzImg[i:i+imfiltersize,j:j+imfiltersize,channel])
    return output
