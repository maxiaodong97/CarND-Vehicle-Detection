**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video_output.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 27 - 46 extract_features of the file called `svm.py`.  It goes through all training samples and extract bin_spatial, color_hist and hog features. Hog features is extracted by calling skimage.feature hog, refer line 24 of `helper.py`. 

I started by reading in all the `vehicle` and `non-vehicle` images. The images source is from https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip and https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=16`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I pick HOG parameters to maximize test accuracy and minimize the time of extract features and prediction, also minimize the false positive.

I found for a 64x64 image, prediction is around 0.0002 seconds (0.002/10), but extract hog features takes relatively long time and really depends on parameters.

I use following as a baseline from course material.

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

This gives pretty good accuracy around 98.5%, but it also takes long time to extract features, around 0.015 seconds per sample. Since later I will use aboud 250 sliding windows (sub images) to search for vehicle. These accuracy will generate %1.5 * 250 ~ 4 "bad" windows,  either false positives or false negatives. With thresholding, 4 false detections is tolerable. I tried to maintain this accuracy but try to reduce the time, since 0.015 * 250 will need 3.75 seconds to process each frame, kind of too long.

So my next goal is to reduce the time to extract feature while not sacraficing too much test accuracy.

1. First I set spatial_feat and hist_feat to false, it will reduce accuracy about 2% but don't save much time. So I decide to keep them on.

2. I tried all color space and train SVM model, and found RGB gives worst test accuracy, while HLS, YUV and YCrCb gives much better test accuracy.  So I decide to choose YCrCb color space. 

3. I found extract hog features is very expensive and CrCb channel don't give much improvements, so I decide to only use Y channel hog features. This gives test accuracy 0.9595 with 4932 features @ 0.0047 seconds to extract, looks optimistic, but accuracy drops too much. So I decides to still use 'ALL' for hog_channel.

4. I tried orientations, larger orientation will create more features and do not increase time to extract hog feature, I tried 9, 13, 16, 20, 30.  It looks like 16 is a good number as more increase go not give better result.

This gives test accuracy 0.962 with 6304 features @ 0.0048 seconds to extract, so I decide to use 16.

5. I try pix_per_cell = 16

This gives test accuracy 0.987 with 4896 features @ 0.0055 seconds to extract features. so I decide to use 16 for pix_per_cell. 

In summay, here is the final parameters that I picked:

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 16  # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off


This gives: 

Test Accuracy of SVC =  0.987
False positive 0.28%
False negative 1.01% 


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using sklearn.svm LinearSVC. Refer line 50-79 in `svm.py`. I hog feature in all 3 YCrCb channel as features to train with car and noncar sampeles. First I extract features from img, then normalize image with StandardScaler to scale to zero mean and unit variance before training the classifier, then I use 90/10 to split samples to training set and test set. Finally I train the model and save the training result to `svm.pkl` file.


In addition, I also train the sample with CNN, refer `cnn.py`, I am able to archive 0.989 test accuracy and it is 4 times faster to processing a frame comparing with SVM HOG, if I want the same level of classification test accuracy, but SVM HOG looks has more parameters to tune and more flexible.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I use sliding_window function line 134 of file `helper.py` (from class code) to create a number of sliding window within an area to do the match, I use 50% overlap to compromize the accuracy and speed. Higher overlap can give more accurate bounding box of the cars. Because I don't have a high performance hardware PC, so I just use 50% overlap to make things faster. 

To capture the small car (far) and larger cars(near), I use 5 scale starting from 80 pix in the far, 160 pix in the near.  ystart and ystop is also adjust to reduce the searching area. please refer line 165 of file `pipeline.py`. 

```python
for scale in [1.0, 1.2, 1.5, 1.7, 2.0]:
        ystart = 400
        ystop = 540 + int(scale * 60)
        if fast_find and method == 'svm':
            bbox_list = find_cars(img, 250, ystart, ystop, scale)
        else:
            xy_window = (int(80 * scale), int(80 * scale))
            windows = slide_window(img, x_start_stop=[250, None], y_start_stop=[
                                   ystart, ystop], xy_window=xy_window, xy_overlap=(0.5, 0.5))
            bbox_list = search_windows(img, windows, method)
        total_bbox_list += bbox_list

```

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 5 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:



![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problem is that I found there are so much hyper parameter to tune to get HOG working nicely. For me, I think CNN is a better choice than SVM HOG. As it tooks me about 2 hour to get CNN producing nice video output, but 2 days to tune those HOG parameters. cv2.imread and mpimg.imread is tricky but was given headsup in the project guide, thanks for that.  

The pipeline will fail in following scenarios: 
1. I only search area in [ 400, 680 ] on y direciton, and [250, None] on x direction. This assumption only works for project video, but not on large slope road or drive on right side of the road.

2. HOG doesn't work well for rotated image, if road is not even between left and right, it may not give good result. Training samples needs to be big enough to cover all case, or it can be augumented. 

3. I used data set of 17760 for training, it doesnt cover enough different angle and type of the vehicles. More samples will help the model to generalize. 
