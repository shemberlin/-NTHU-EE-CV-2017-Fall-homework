# Before trying to construct hybrid images, it is suggested that you
# implement my_imfilter.m and then debug it using proj1_test_filtering.py

from my_imfilter import my_imfilter
from vis_hybrid_image import vis_hybrid_image
from normalize import normalize
from gauss2D import gauss2D
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import scipy
import os

''' Setup '''
# read images and convert to floating point format
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image1 = mpimg.imread(main_path + '/data/dog.bmp')
a=image1.shape[0]*image1.shape[1]
image2 = mpimg.imread(main_path + '/data/cat.bmp')

image3 = mpimg.imread(main_path + '/data/marilyn.bmp')
image4 = mpimg.imread(main_path + '/data/einstein.bmp')
b=image3.shape[0]*image3.shape[1]

image6 = mpimg.imread(main_path + '/data/submarine.bmp')
image5 = mpimg.imread(main_path + '/data/fish.bmp')
c=image5.shape[0]*image5.shape[1]

image7 = mpimg.imread(main_path + '/data/bird.bmp')
d=image7.shape[0]*image7.shape[1]
image8 = mpimg.imread(main_path + '/data/plane.bmp')

image9 = mpimg.imread(main_path + '/data/bicycle.bmp')
image10 = mpimg.imread(main_path + '/data/motorcycle.bmp')
e=image9.shape[0]*image9.shape[1]

image12 = mpimg.imread(main_path + '/data/baozi.jpg')
image11 = mpimg.imread(main_path + '/data/Xi Jinping.jpg')

image1 = image1.astype(np.single)/255
image2 = image2.astype(np.single)/255
image3 = image3.astype(np.single)/255
image4 = image4.astype(np.single)/255
image5 = image5.astype(np.single)/255
image6 = image6.astype(np.single)/255
image7 = image7.astype(np.single)/255
image8 = image8.astype(np.single)/255
image9 = image9.astype(np.single)/255
image10 = image10.astype(np.single)/255
image11 = image11.astype(np.single)/255
image12 = image12.astype(np.single)/255

# Several additional test cases are provided for you, but feel free to make
# your own (you'll need to align the images in a photo editor such as
# Photoshop). The hybrid images will differ depending on which image you
# assign as image1 (which will provide the low frequencies) and which image
# you asign as image2 (which will provide the high frequencies)

''' Filtering and Hybrid Image construction '''
cutoff_frequency = 7
cutoff_frequency_2 = 4 
cutoff_frequency_3 = 6 
cutoff_frequency_4 = 5.5 
cutoff_frequency_5 = 5.5
cutoff_frequency_6 = 4.5
# This is the standard deviation, in pixels, of the 
# Gaussian blur that will remove the high frequencies from one image and 
# remove the low frequencies from another image (by subtracting a blurred
# version from the original version). You will want to tune this for every
# image pair to get the best results.
gaussian_filter = gauss2D(shape=(cutoff_frequency*4+1,cutoff_frequency*4+1), sigma = cutoff_frequency)
gaussian_filter_2 = gauss2D(shape=(cutoff_frequency_2*4+1,cutoff_frequency_2*4+1), sigma = cutoff_frequency_2)
gaussian_filter_3 = gauss2D(shape=(cutoff_frequency_3*4+1,cutoff_frequency_3*4+1), sigma = cutoff_frequency_3)
gaussian_filter_4 = gauss2D(shape=(cutoff_frequency_4*4+1,cutoff_frequency_4*4+1), sigma = cutoff_frequency_4)
gaussian_filter_5 = gauss2D(shape=(cutoff_frequency_5*4+1,cutoff_frequency_5*4+1), sigma = cutoff_frequency_5)
gaussian_filter_6 = gauss2D(shape=(cutoff_frequency_6*4+1,cutoff_frequency_6*4+1), sigma = cutoff_frequency_6)
#########################################################################
# TODO: Use my_imfilter create 'low_frequencies' and                    #
# 'high_frequencies' and then combine them to create 'hybrid_image'     #
#########################################################################
#########################################################################
# Remove the high frequencies from image1 by blurring it. The amount of #
# blur that works best will vary with different image pairs             #
#########################################################################
low_frequencies = my_imfilter(image1, gaussian_filter)
low_frequencies_2 = my_imfilter(image3, gaussian_filter_2)
low_frequencies_3 = my_imfilter(image5, gaussian_filter_3)
low_frequencies_4 = my_imfilter(image7, gaussian_filter_4)
low_frequencies_5 = my_imfilter(image9, gaussian_filter_5)
low_frequencies_6 = my_imfilter(image11, gaussian_filter_6)
############################################################################
# Remove the low frequencies from image2. The easiest way to do this is to #
# subtract a blurred version of image2 from the original version of image2.#
# This will give you an image centered at zero with negative values.       #
############################################################################
high_frequencies = image2 - my_imfilter(image2, gaussian_filter)
high_frequencies_2 = image4 - my_imfilter(image4, gaussian_filter_2)
high_frequencies_3 = image6 - my_imfilter(image6, gaussian_filter_3)
high_frequencies_4 = image8 - my_imfilter(image8, gaussian_filter_4)
high_frequencies_5 = image10 - my_imfilter(image10, gaussian_filter_5)
high_frequencies_6 = image12 - my_imfilter(image12, gaussian_filter_6)

############################################################################
# Combine the high frequencies and low frequencies                         #
############################################################################
hybrid_image = normalize(low_frequencies + high_frequencies)
hybrid_image_2 = normalize(low_frequencies_2 + high_frequencies_2)
hybrid_image_3 = normalize(low_frequencies_3 + high_frequencies_3)
hybrid_image_4 = normalize(low_frequencies_4 + high_frequencies_4)
hybrid_image_5 = normalize(low_frequencies_5 + high_frequencies_5)
hybrid_image_6 = normalize(low_frequencies_6 + high_frequencies_6)
''' Visualize and save outputs '''
print("Cat and Dog")
print("Cutoff frequency",cutoff_frequency)
plt.figure(1)
plt.imshow(low_frequencies)
plt.figure(2)
plt.imshow(high_frequencies+0.5)
vis = vis_hybrid_image(hybrid_image)
plt.figure(3)
plt.imshow(vis)
plt.imsave(main_path+'/results/low_frequencies.png', low_frequencies, 'quality', 95)
plt.imsave(main_path+'/results/high_frequencies.png', high_frequencies + 0.5, 'quality', 95)
plt.imsave(main_path+'/results/hybrid_image.png', hybrid_image, 'quality', 95)
plt.imsave(main_path+'/results/hybrid_image_scales.png', vis, 'quality', 95)
plt.show()
print("Marilyn and Einstein")
print("Cutoff frequency",cutoff_frequency_2)
plt.figure(4)
plt.imshow(low_frequencies_2)
plt.figure(5)
plt.imshow(high_frequencies_2+0.5)
vis_2 = vis_hybrid_image(hybrid_image_2)
plt.figure(6)
plt.imshow(vis_2)
plt.imsave(main_path+'/results/low_frequencies_2.png', low_frequencies_2, 'quality', 95)
plt.imsave(main_path+'/results/high_frequencies_2.png', high_frequencies_2 + 0.5, 'quality', 95)
plt.imsave(main_path+'/results/hybrid_image_2.png', hybrid_image_2, 'quality', 95)
plt.imsave(main_path+'/results/hybrid_image_scales_2.png', vis_2, 'quality', 95)
plt.show()
print("Submarine and Fish")
print("Cutoff frequency",cutoff_frequency_3)
plt.figure(7)
plt.imshow(low_frequencies_3)
plt.figure(8)
plt.imshow(high_frequencies_3+0.5)
vis_3 = vis_hybrid_image(hybrid_image_3)
plt.figure(9)
plt.imshow(vis_3)
plt.imsave(main_path+'/results/low_frequencies_3.png', low_frequencies_3, 'quality', 95)
plt.imsave(main_path+'/results/high_frequencies_3.png', high_frequencies_3 + 0.5, 'quality', 95)
plt.imsave(main_path+'/results/hybrid_image_3.png', hybrid_image_3, 'quality', 95)
plt.imsave(main_path+'/results/hybrid_image_scales_3.png', vis_3, 'quality', 95)
plt.show()
print("Bird and Plane")
print("Cutoff frequency",cutoff_frequency_4)
plt.figure(10)
plt.imshow(low_frequencies_4)
plt.figure(11)
plt.imshow(high_frequencies_4+0.5)
vis_4 = vis_hybrid_image(hybrid_image_4)
plt.figure(12)
plt.imshow(vis_4)
plt.imsave(main_path+'/results/low_frequencies_4.png', low_frequencies_4, 'quality', 95)
plt.imsave(main_path+'/results/high_frequencies_4.png', high_frequencies_4 + 0.5, 'quality', 95)
plt.imsave(main_path+'/results/hybrid_image_4.png', hybrid_image_4, 'quality', 95)
plt.imsave(main_path+'/results/hybrid_image_scales_4.png', vis_4, 'quality', 95)
plt.show()
print("Bicycle and Motorcycle")
print("Cutoff frequency",cutoff_frequency_5)
plt.figure(13)
plt.imshow(low_frequencies_5)
plt.figure(14)
plt.imshow(high_frequencies_5+0.5)
vis_5 = vis_hybrid_image(hybrid_image_5)
plt.figure(15)
plt.imshow(vis_5)
plt.imsave(main_path+'/results/low_frequencies_5.png', low_frequencies_5, 'quality', 95)
plt.imsave(main_path+'/results/high_frequencies_5.png', high_frequencies_5 + 0.5, 'quality', 95)
plt.imsave(main_path+'/results/hybrid_image_5.png', hybrid_image_5, 'quality', 95)
plt.imsave(main_path+'/results/hybrid_image_scales_5.png', vis_5, 'quality', 95)
plt.show()
print("習近平與小熊維尼")
print("Cutoff frequency",cutoff_frequency_6)
plt.figure(16)
plt.imshow(low_frequencies_6)
plt.figure(17)
plt.imshow(high_frequencies_6+0.5)
vis_6 = vis_hybrid_image(hybrid_image_6)
plt.figure(18)
plt.imshow(vis_6)
plt.imsave(main_path+'/results/low_frequencies_6.png', low_frequencies_6, 'quality', 95)
plt.imsave(main_path+'/results/high_frequencies_6.png', high_frequencies_6 + 0.5, 'quality', 95)
plt.imsave(main_path+'/results/hybrid_image_6.png', hybrid_image_6, 'quality', 95)
plt.imsave(main_path+'/results/hybrid_image_scales_6.png', vis_6, 'quality', 95)
plt.show()








