# 105061592 林鉉博
# Homework 1 / Image Filtering and Hybrid Images

## Overview
Write an image filtering function and use it to create hybrid images.Using the Gaussian filter as a low frequency filter.After knowing low frequency . The original image minus the low frequency is equal to the high frequency.
```sh
low_frequencies = my_imfilter(image1, gaussian_filter)
high_frequencies = image1- my_imfilter(image1, gaussian_filter)
hybrid_image = normalize(low_frequencies + high_frequencies)
```
## Installation
- Main function:proj1,my_imfilter
- Other required packages: numpy, scipy, matplotlib, pytictoc, os.
- Functions: gauss2D, normalize, vis_hybrid_image, proj1_test_filtering,

## Implementation
### proj1.py 
- Write the tips given by the teacher.
    ```sh       
    low_frequencies = my_imfilter(image1, gaussian_filter)
    high_frequencies = image1- my_imfilter(image1, gaussian_filter)
    hybrid_image = normalize(low_frequencies + high_frequencies)
    ```
- After several executions, it is found that these cutoff frequencies give the best results.
```sh
cutoff_frequency = 7
cutoff_frequency_2 = 4 
cutoff_frequency_3 = 6 
cutoff_frequency_4 = 5.5 
cutoff_frequency_5 = 5.5
cutoff_frequency_6 = 4.5
```
### my_imfilter.py
##### Zero padding for the image 
- In order to create a zero matrix with size of the original image plus the filter. First find the filter midpoint,and then use np.pad this function can automatically fill 0.
- After running the program, we will get the original image (matrix) with the surrounding zero matrix.(the original image plus the filter)
- Such a matrix lets us successfully do the filter convolution.
    ```sh
    (width,height)=(int(image.shape[0]),int(image.shape[1]))#image size
    median=int((imfilter.shape[0]-1)/2)#filter mid
    udzeros=int(imfilter.shape[0]-1)#up down fill 0
    rgb=range(image.shape[2])#012
    filterm = (median, median)
    AzImg=np.zeros(((width+udzeros),(height+udzeros),3))
    for channel in rgb:        
        AzImg[:,:,channel] = np.pad(image[:,:,channel], pad_width=filterm, mode='constant', constant_values=0)
    ```
##### Convolution
- Convolution kernel around its core elements rotated 180 degrees clockwise.
    ```sh
    A = np.flip(imfilter,1)#Flip an array horizontally (axis=1).
    B = np.flip(A,0)#Flip an array vertically (axis=0).filter 180 degree rotation
    ```
- Get imfilter's size and create a zero matrix(in order to let output[i,j,channel] can work)
    ```sh
    imfiltersize=imfilter.shape[0]
    output = np.zeros_like(image)
  ```
- Use forloop to move the center element of the convolution kernel so that it is located just above the pixel to be processed (in fact, the position of the pixel to be processed, where the upper part refers to the two matrices that overlap and the elements are Processing elements overlap).
-  In the rotated convolution kernel, the pixel value of the input image is multiplied by the weight.
- The result of the third step is the output pixel corresponding to the input pixel.
    ```sh
    for channel in rgb:#0~2
        for i in range(width):
            for j in range(height):
                output[i,j,channel] = np.sum(B * AzImg[i:i+imfiltersize,j:j+imfiltersize,channel])
    return output
   ```
- Convolution follow this GIF:
- White area:pad 0
- Blue area:original image
    - White area +Blue area:AzImg(pad 0+original image)
- Grey area:imfilter
- Green area:output
<img src="https://github.com/shemberlin/homework1/blob/master/index_files/convolution.gif" width="247" height="271">

# Result
##### Cutoff_frequency = 7
  |![image](https://github.com/shemberlin/homework1/blob/master/results/low_frequencies.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/high_frequencies.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/hybrid_image.png)|![image](https://github.com/shemberlin/homework1/blob/master/results/hybrid_image_scales.png) |
  | ------------- | ------------- | -------------| -------------|
  | Low Frequency image  | High Frequency Image  | Hybrid Image | Scale Image |
##### Cutoff_frequency_2 = 4
  |![image](https://github.com/shemberlin/homework1/blob/master/results/low_frequencies_2.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/high_frequencies_2.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/hybrid_image_2.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/hybrid_image_scales_2.png) |
  | ------------- | ------------- | -------------| -------------|
  | Low Frequency image  | High Frequency Image  | Hybrid Image | Scale Image |
##### Cutoff_frequency_3 = 6
  |![image](https://github.com/shemberlin/homework1/blob/master/results/low_frequencies_3.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/high_frequencies_3.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/hybrid_image_3.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/hybrid_image_scales_3.png) |
  | ------------- | ------------- | -------------| -------------|
  | Low Frequency image  | High Frequency Image  | Hybrid Image | Scale Image |

##### Cutoff_frequency_4 = 5.5
  |![image](https://github.com/shemberlin/homework1/blob/master/results/low_frequencies_4.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/high_frequencies_4.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/hybrid_image_4.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/hybrid_image_scales_4.png) |
  | ------------- | ------------- | -------------| -------------|
  | Low Frequency image  | High Frequency Image  | Hybrid Image | Scale Image |
##### Cutoff_frequency_5 = 5.5
  |![image](https://github.com/shemberlin/homework1/blob/master/results/low_frequencies_5.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/high_frequencies_5.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/hybrid_image_5.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/hybrid_image_scales_5.png) |
  | ------------- | ------------- | -------------| -------------|
  | Low Frequency image  | High Frequency Image  | Hybrid Image | Scale Image |
  
##### Cutoff_frequency_6 = 4.5
  |![image](https://github.com/shemberlin/homework1/blob/master/results/low_frequencies_6.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/high_frequencies_6.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/hybrid_image_6.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/hybrid_image_scales_6.png) |
  | ------------- | ------------- | -------------| -------------|
  | Low Frequency image  | High Frequency Image  | Hybrid Image | Scale Image |
   
##### Filter testing
  |![image](https://github.com/shemberlin/homework1/blob/master/data/cat.bmp) |![image](https://github.com/shemberlin/homework1/blob/master/results/test/blur_image.png) |![image](https://github.com/shemberlin/homework1/blob/master/results//test/large_blur_image.png) |
  | ------------- | ------------- | -------------|
  | Original Iamge | Blur Image  | Large blur Image |
   |![image](https://github.com/shemberlin/homework1/blob/master/results/test/laplacian_image.png) |![image](https://github.com/shemberlin/homework1/blob/master/results/test/sobel_image.png) |![image](https://github.com/shemberlin/homework1/blob/master/results//test/high_pass_image.png) |
  | Laplacian Image | Sobel Image  | High Pass Image |
