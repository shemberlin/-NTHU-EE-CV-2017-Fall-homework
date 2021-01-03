# 105061592 林鉉博
# Project 2 / Panorama Stitching

## Overview
Stitch multiple images together under a simplified case of real-world scenario.(In order image)
```sh
1. Get SIFT points and descriptors(using vlfeat)
2. Matching SIFT Descriptors
3. Fitting the Transformation Matrix
4. RANSAC
5. Stitching Multiple Images(a. Stitching ordered sequence of images)
```
## Installation
- Test function:StitchTester,TransformationTester
- Evaluate function:EvaluateAffineMatrix,EvaluateSIFTMatcher
- Other required packages: numpy, scipy, matplotlib, pytictoc, os,sys
- Functions: ComputeAffineMatrix,MultipleStitch,PairStitch,RANSACFit,SIFTSimpleMatcher
- Install cyvlfeat for fetching sift features

## Implementation
### SIFTSimpleMatcher.py 
##### Get SIFT points and descriptors
- First create an empty matrix.
- Then use the np.linalg.norm in the second loop to calculate the Euclidean distance between descriptor1 and descriptor2.
- After getting the new value(Euclidean distance), add a new value to the list using append.
- Jump out of the variable j for loop, back to i loop.
- Use np.argsort to get the minimum Euclidean distance index.
- At this time, if the minimum Euclidean distance = THRESH * second minimum Euclidean distance is satisfied, the matched value will be put into the listmatch.
- And then temp will be set back empty matrix until finish all loop.
```sh       
     listmatch = [] 
    for i in range(len(descriptor1)):
        temp=[]
        for j in range(len(descriptor2)):
            distance = np.linalg.norm(descriptor1[i] - descriptor2[j])
            temp.append(distance) 

        index = np.argsort(temp)
        if temp[index[0]] < THRESH * temp[index[1]]:
            listmatch.append([i, index[0]])
            
    match = np.array(listmatch)#21*2 
```

### ComputeAffineMatrix.py
##### Fitting the Transformation Matrix 
- H*P1=P2 to yield P1'*H'=P2'. Then PYTHON can solve for H'( H = (inv(P1P1') * (P1P2'))')
- Use np.matmu do Matrix multiplication
- Use np.linalg.inv do Inverse matrix
    ```sh
    A=np.matmul(P1, P1.T)
    n=np.matmul(P1, P2.T)
    m=np.linalg.inv(A)
    H = np.matmul(m, n).T
    ```
### RANSACFit.py
##### RANSAC
- Compute the error using transformation matrix H to transform the point in pt1 to its matching point in pt2.
- Use np.Concatenate () more efficient 
- Find the most similar transformation matrix H using Euclidean np.linalg.norm
    ```sh
    N=len(match)
    array1=pt1[match[:, 0]].T
    array2=pt2[match[:, 1]].T
    onearray=np.ones([1,N])
       
    P1 = np.concatenate([array1,onearray])
    P1 = np.matmul(H, P1)
    P2 = np.concatenate([array2,onearray])
              
    dists = np.linalg.norm(P1 - P2,axis=0).T
    ```
### MultipleStitch.py
##### Stitching Multiple Images(a. Stitching ordered sequence of images)
- This function stitches multiple Images together and outputs the panoramic stitched image with a chain of input Images and its corresponding Transformations. 
- Compute the transform matrix for i-th frame to the reference frame
- If the current frame index is less than the reference frame index,the i_To_iPlusOne_Transform matrix is multiplied by the T matrix
- If the current frame index is greater than the reference frame index,the i_To_iPlusOne_Transform matrix is multiplied by the T matrix
- Finally, the use of inverse conversion.
   ```sh
    T = np.identity(3)
    if currentFrameIndex < refFrameIndex: 
        for i in range(currentFrameIndex, refFrameIndex):
            T = np.matmul(i_To_iPlusOne_Transform[i], T)
    elif currentFrameIndex > refFrameIndex:
        for i in range(currentFrameIndex, refFrameIndex, -1):
            T = np.matmul(np.linalg.inv(i_To_iPlusOne_Transform[i-1]), T)
    ```

# Result
### Original image:Rainier
  |![image](https://github.com/shemberlin/homework2/blob/master/data/Rainier1.png) |![image](https://github.com/shemberlin/homework2/blob/master/data/Rainier2.png) |![image](https://github.com/shemberlin/homework2/blob/master/data/Rainier3.png)|![image](https://github.com/shemberlin/homework2/blob/master/data/Rainier4.png) |![image](https://github.com/shemberlin/homework2/blob/master/data/Rainier5.png) |![image](https://github.com/shemberlin/homework2/blob/master/data/Rainier6.png) |
  | ------------- | ------------- | -------------| -------------|-------------|-------------|
### Original image:Yosemite
 |![image](https://github.com/shemberlin/homework2/blob/master/data/yosemite1.jpg) |![image](https://github.com/shemberlin/homework2/blob/master/data/yosemite2.jpg) |![image](https://github.com/shemberlin/homework2/blob/master/data/yosemite3.jpg)|![image](https://github.com/shemberlin/homework2/blob/master/data/yosemite4.jpg) |
 | ------------- | ------------- | -------------| -------------|
### Original image:Uttower
 |![image](https://github.com/shemberlin/homework2/blob/master/data/uttower1.jpg) |![image](https://github.com/shemberlin/homework2/blob/master/data/uttower2.jpg) |
  | ------------- | ------------- |
### Original image:Hanging
 |![image](https://github.com/shemberlin/homework2/blob/master/data/Hanging1.png) |![image](https://github.com/shemberlin/homework2/blob/master/data/Hanging2.png) |
  | ------------- | ------------- |
### Original image:MelakwaLake
 |![image](https://github.com/shemberlin/homework2/blob/master/data/MelakwaLake1.png) |![image](https://github.com/shemberlin/homework2/blob/master/data/MelakwaLake2.png) |
  | ------------- | ------------- |
### Rainier_Pano
|![image](https://github.com/shemberlin/homework2/blob/master/results/pano.jpg) |
 | ------------- |
### Yosemite_Pano
|![image](https://github.com/shemberlin/homework2/blob/master/results/pano2.jpg) |
 | ------------- |

### Uttower_Pano
|![image](https://github.com/shemberlin/homework2/blob/master/results/uttower_pano.jpg) |
 | ------------- |
 
### Hanging_pano
|![image](https://github.com/shemberlin/homework2/blob/master/results/Hanging_pano.png) |
 | ------------- |
 
### MelakwaLake_pano
|![image](https://github.com/shemberlin/homework2/blob/master/results/MelakwaLake_pano.png) |
 | ------------- |
