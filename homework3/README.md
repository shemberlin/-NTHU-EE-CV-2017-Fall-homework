# 105061592 林鉉博
# Project 3 / Scene recognition with bag of words

## Overview
The purpose of HW3 is to let us understand the CV in the image recognition.
1.Image representations
2.Classification
```sh
tiny images + nearest neighbor
bag of SIFT + nearest neighbor
bag of SIFT + linear SVM
```


## Implementation
### Image representations
##### get_tiny_images.py 
- First resize the original image to a very small squre resolution, using a 16x16 based on the teacher's instructions.
- Then convert the matrix to a 256-dimensional array, .
- Subtract the mean from the mean, then remove vareiance 
- Mean be 0 and vareiance be 1
- The overall performance will improve.

```sh       
    N = len(image_paths)
    tiny_images = np.zeros((N,256))
    
    for i in range(N):
        img = Image.open(image_paths[i])
        img = img.resize(16,16)
        img = np.array(img)
        tiny_images[i,:] = img.flatten()
        tiny_images[i,:] = tiny_images[i,:] - np.mean(tiny_images[i,:])
        tiny_images[i,:] = tiny_images[i,:]/np.std(tiny_images[i,:])
```

##### build_vocablary.py
 
- From the training images, execute sample SIFTdescriptors, then cluster these descriptors in k-means.
- Find a good representative of the point.
- Return cluster's center.
    ```sh
    bag_of_features = []
    print("Extract SIFT features")
    for path in image_paths:
        img = np.asarray(Image.open(path),dtype='float32')
        frames, descriptors = dsift(img, step, fast=True)
        bag_of_features.append(descriptors)
        
    bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')
    print("Compute vocab")
    start_t = time()
    vocab = kmeans(bag_of_features, vocab_size, initialization="PLUSPLUS")        
    end_t = time()
    print("It takes ", (start_t - end_t), " to compute vocab.")
    ```
##### get_bags_of_sifts.py
- Find each local feature nearest to his cluster.
- Create a histogram to see the status of each cluster that is grouped. 
- Normalize histogram(compare the value of each bag of feature).
    ```sh
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
    ```
### Classification
##### nearest_neighbor_classify.py

- This function is used to know which category the test image is closest to.
- k=10
    ```sh
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
    ```
##### svm_classify.py

- This function stitches multiple Images 
- Use scikit learn function
- Change C=0.01,0.1,1,10,100,1000
   ```sh
    c2 = LinearSVC(C = C)
    c2.fit(train_image_feats, train_labels)
    pred_label = c2.predict(test_image_feats)
    ```

# Result

<img src="1.PNG" width=80% height=80%>
<img src="2.PNG" width=80% height=80%>
<img src="confusion_matrix.png" width=40% height=40%>








 <table border=0 cellpadding=4 cellspacing=1>
 <tr>
 <th>Category name</th>
 <th>Accuracy</th>
 <th >Sample training images</th>
 <th >Sample true positives</th>
 <th >False positives with true label</th>
 <th >False negatives with wrong predicted label</th>
  </tr>
  <tr>
  <td>Kitchen</td>
  <td>52%</td>
  <td bgcolor=LightBlue><img src="thumbnails/Kitchen_train_image_0001.jpg" width=100 height=75></td>
  <td bgcolor=LightGreen><img src="thumbnails/Kitchen_TP_image_0192.jpg" width=100 height=75></td>
  <td bgcolor=LightCoral><img src="thumbnails/Kitchen_FP_image_0077.jpg" width=100 height=75></td>
  <td bgcolor=#FFBB55><img src="thumbnails/Kitchen_FN_image_0190.jpg" width=57 height=75></td>
  </tr>
  <tr>
  <td>Store</td>
  <td>54%</td>
  <td bgcolor=LightBlue><img src="thumbnails/Store_train_image_0001.jpg" width=112 height=75></td>
  <td bgcolor=LightGreen><img src="thumbnails/Store_TP_image_0150.jpg" width=74 height=75></td>
  <td bgcolor=LightCoral><img src="thumbnails/Store_FP_image_0021.jpg" width=75 height=75></td>
  <td bgcolor=#FFBB55><img src="thumbnails/Store_FN_image_0151.jpg" width=100 height=75></td>
  </tr>
  <tr>
  <td>Bedroom</td>
  <td>47%</td>
  <td bgcolor=LightBlue><img src="thumbnails/Bedroom_train_image_0001.jpg" width=100 height=75></td>
  <td bgcolor=LightGreen><img src="thumbnails/Bedroom_TP_image_0180.jpg" width=57 height=75></td>
  <td bgcolor=LightCoral><img src="thumbnails/Bedroom_FP_image_0130.jpg" width=100 height=75></td>
  <td bgcolor=#FFBB55><img src="thumbnails/Bedroom_FN_image_0176.jpg" width=100 height=75></td>
  </tr>
  <tr>
  <td>LivingRoom</td>
  <td>24%</td>
   <td bgcolor=LightBlue><img src="thumbnails/LivingRoom_train_image_0001.jpg" width=114 height=75></td>
  <td bgcolor=LightGreen><img src="thumbnails/LivingRoom_TP_image_0138.jpg" width=100 height=75></td>
  <td bgcolor=LightCoral><img src="thumbnails/LivingRoom_FP_image_0010.jpg" width=134 height=75></td>
  <td bgcolor=#FFBB55><img src="thumbnails/LivingRoom_FN_image_0147.jpg" width=113 height=75></td>
  </tr>
  <tr>
  <td>Office</td>
  <td>92%</td>
  <td bgcolor=LightBlue><img src="thumbnails/Office_train_image_0002.jpg" width=94 height=75></td>
  <td bgcolor=LightGreen><img src="thumbnails/Office_TP_image_0185.jpg" width=103 height=75></td>
  <td bgcolor=LightCoral><img src="thumbnails/Office_FP_image_0047.jpg" width=75 height=75></td>
  <td bgcolor=#FFBB55><img src="thumbnails/Office_FN_image_0180.jpg" width=102 height=75></td>
  </tr>
  <tr>
  <td>Industrial</td>
  <td>37%</td>
  <td bgcolor=LightBlue><img src="thumbnails/Industrial_train_image_0002.jpg" width=75 height=75></td>
  <td bgcolor=LightGreen><img src="thumbnails/Industrial_TP_image_0148.jpg" width=75 height=75></td>
  <td bgcolor=LightCoral><img src="thumbnails/Industrial_FP_image_0021.jpg" width=75 height=75></td>
  <td bgcolor=#FFBB55><img src="thumbnails/Industrial_FN_image_0152.jpg" width=75 height=75></td>
  </tr>
  <tr>
  <td>Suburb</td>
  <td>91%</td>
  <td bgcolor=LightBlue><img src="thumbnails/Suburb_train_image_0002.jpg" width=113 height=75></td>
  <td bgcolor=LightGreen><img src="thumbnails/Suburb_TP_image_0176.jpg" width=113 height=75></td>
  <td bgcolor=LightCoral><img src="thumbnails/Suburb_FP_image_0099.jpg" width=107 height=75></td>
  <td bgcolor=#FFBB55><img src="thumbnails/Suburb_FN_image_0175.jpg" width=113 height=75></td>
  </tr>
  <tr>
  <td>InsideCity</td>
  <td>57%</td>
  <td bgcolor=LightBlue><img src="thumbnails/InsideCity_train_image_0005.jpg" width=75 height=75></td>
  <td bgcolor=LightGreen><img src="thumbnails/InsideCity_TP_image_0137.jpg" width=75 height=75></td>
  <td bgcolor=LightCoral><img src="thumbnails/InsideCity_FP_image_0086.jpg" width=75 height=75></td>
  <td bgcolor=#FFBB55><img src="thumbnails/InsideCity_FN_image_0140.jpg" width=75 height=75></td>
  </tr>
  <tr>
  <td>TallBuilding</td>
  <td>66%</td>
  <td bgcolor=LightBlue><img src="thumbnails/TallBuilding_train_image_0010.jpg" width=75 height=75></td>
  <td bgcolor=LightGreen><img src="thumbnails/TallBuilding_TP_image_0131.jpg" width=75 height=75></td>
  <td bgcolor=LightCoral><img src="thumbnails/TallBuilding_FP_image_0005.jpg" width=113 height=75></td>
  <td bgcolor=#FFBB55><img src="thumbnails/TallBuilding_FN_image_0128.jpg" width=75 height=75></td>
  </tr>
  <tr>
  <td>Street</td>
  <td>58%</td>
  <td bgcolor=LightBlue><img src="thumbnails/Street_train_image_0001.jpg" width=75 height=75></td>
  <td bgcolor=LightGreen><img src="thumbnails/Street_TP_image_0147.jpg" width=75 height=75></td>
  <td bgcolor=LightCoral><img src="thumbnails/Street_FP_image_0156.jpg" width=75 height=75></td>
  <td bgcolor=#FFBB55><img src="thumbnails/Street_FN_image_0149.jpg" width=75 height=75></td>
  </tr>
  <tr>
  <td>Highway</td>
 <td>69%</td>
  <td bgcolor=LightBlue><img src="thumbnails/Highway_train_image_0009.jpg" width=75 height=75></td>
  <td bgcolor=LightGreen><img src="thumbnails/Highway_TP_image_0162.jpg" width=75 height=75></td>
  <td bgcolor=LightCoral><img src="thumbnails/Highway_FP_image_0103.jpg" width=75 height=75></td>
  <td bgcolor=#FFBB55><img src="thumbnails/Highway_FN_image_0157.jpg" width=75 height=75></td>
  </tr>
  <tr>
  <td>OpenCountry</td>
  <td>47%</td>
  <td bgcolor=LightBlue><img src="thumbnails/OpenCountry_train_image_0003.jpg" width=75 height=75></td>
  <td bgcolor=LightGreen><img src="thumbnails/OpenCountry_TP_image_0123.jpg" width=75 height=75></td>
  <td bgcolor=LightCoral><img src="thumbnails/OpenCountry_FP_image_0040.jpg" width=75 height=75></td>
  <td bgcolor=#FFBB55><img src="thumbnails/OpenCountry_FN_image_0125.jpg" width=75 height=75></td>
  </tr>
  <tr>
  <td>Coast</td>
  <td>77%</td>
  <td bgcolor=LightBlue><img src="thumbnails/Coast_train_image_0006.jpg" width=75 height=75></td>
  <td bgcolor=LightGreen><img src="thumbnails/Coast_TP_image_0129.jpg" width=75 height=75></td>
  <td bgcolor=LightCoral><img src="thumbnails/Coast_FP_image_0117.jpg" width=78 height=75></td>
  <td bgcolor=#FFBB55><img src="thumbnails/Coast_FN_image_0130.jpg" width=75 height=75></td>
  </tr>
  <tr>
  <td>Mountain</td>
  <td>80%</td>
  <td bgcolor=LightBlue><img src="thumbnails/Mountain_train_image_0002.jpg" width=75 height=75></td>
  <td bgcolor=LightGreen><img src="thumbnails/Mountain_TP_image_0123.jpg" width=75 height=75></td>
  <td bgcolor=LightCoral><img src="thumbnails/Mountain_FP_image_0056.jpg" width=75 height=75></td>
  <td bgcolor=#FFBB55><img src="thumbnails/Mountain_FN_image_0117.jpg" width=75 height=75></td>
  </tr>
  <tr>
  <td>Forest</td>
  <td>94%</td>
  <td bgcolor=LightBlue><img src="thumbnails/Forest_train_image_0003.jpg" width=75 height=75></td>
  <td bgcolor=LightGreen><img src="thumbnails/Forest_TP_image_0142.jpg" width=75 height=75></td>
  <td bgcolor=LightCoral><img src="thumbnails/Forest_FP_image_0107.jpg" width=100 height=75></td>
 <td bgcolor=#FFBB55><img src="thumbnails/Forest_FN_image_0099.jpg" width=75 height=75></td>
  </table>
