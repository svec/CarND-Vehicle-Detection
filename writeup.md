**Vehicle Detection Project Writeup**

**For Christopher Svec**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_nocar.png
[image2]: ./output_images/hogs.png
[runs_plot]: ./output_images/svm-runs.png
[image3]: ./output_images/sliding_windows.png
[test1]: ./output_images/test1.png
[test2]: ./output_images/test2.png
[test3]: ./output_images/test3.png
[test4]: ./output_images/test4.png
[test5]: ./output_images/test5.png
[test6]: ./output_images/test6.png
[heatmap1]: ./output_images/video-40-05.png
[heatmap2]: ./output_images/video-40-06.png
[heatmap3]: ./output_images/video-40-07.png
[heat_final]: ./output_images/video-40-07-final.png

[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

All code and line numbers refer to the project file final_project.py.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is mostly in the functions setup_and_train_classifier()
starting on line 263, which calls the function one_hog() starting on line 339.

I started by reading in all the `vehicle` and `non-vehicle` images as provided
by Udacity.  Here is an example of some of each of the `vehicle` and
`non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the RGB grayscale color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I ran many training runs (126) across different color space, orientation,
pixels_per_cell, and cells_per_block using the SVM described in the
next section. The search space was:
```
        colorspaces = ['LUV', 'YUV', 'RGB', 'HSV', 'HLS', 'YCrCb']
        orientations = [6, 7, 8, 9, 10, 11, 12]
        pixels_per_cell = [8, 16, 32]
        cells_per_block = [2]
        hog_channels = ["ALL"]

```

I sorted the runs by accuracy, and then chose the parameters that yielded a high accuracy, >98%, that trained well.
Some of the high accuracy runs did not seem to produce good classification
results when run on real images, so I used trial and error to pick the parameters
that worked. Here you can see the top runs from the 126 parameter runs, with the one I chose highlighted: YCrCb, orientation=9, pix_per_cell=8, cells_per_block=2, all channels.

![alt text][runs_plot]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM (class LinearSVC) using the one_hog() function mentioned earlier.
one_hog() calls extract_features() which actually extracts the HOG features
from the images. I shuffled the car and not-car data using train_test_split(),
and then trained the SVM using the LinearSVC.fit() function.

I tested its accuracy using 20% of the shuffled input data set using the LinearSVC().score() function.

The HOG features performed well enough that I did not attempt to add other features.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the Udacity-provided find_cars() function, starting at line 401, to implement the sliding window search.

The process_image() function sets up a list of calls to find_cars() using three different scales (1.0, 1.5, and 2.0) and several window regions.

The scale and window regions were determined by a lot of trial and error to find a mix that ran quickly enough, and also identified vehicles well without too many false positives.

Here's an example of all of the search regions:

![alt text][image3]

It's a bit messy, but you can see the three sizes of windows at different starting y values.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here is how my pipeline processed the sample images:

![alt text][test1]
![alt text][test2]
![alt text][test3]
![alt text][test4]
![alt text][test5]
![alt text][test6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_processed.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

My basic single-image pipeline worked well, but there were some false
positives. I filtered these and combined overlapping bounding boxes by using a heatmap (as suggested by Udacity).

On line 627 of process_image() I collect the list of rectangles that matched a vehicle from the current frame.

I insert that list into the `g_vehicle_rects` object, which collects the rectangles from the last 3 frames.

These rectangles were combined to make a heatmap, and then the heatmap was thresholded to find vehicles. Individual vehicles were found using the scipy label function. My pipeline then drew bounding boxes around each labeled vehicle location.

### Here are three frames and their corresponding heatmaps:

![alt_text][heatmap1]
![alt_test][heatmap2]
![alt_test][heatmap3]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt_test][heat_final]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline behaved reasonably well. It found most vehicles in each frame, and eliminated most false positives. 

It had trouble finding cars entering the frame and leaving the searched region of the frame. I would expect it to have problems
with roads and conditions with varying colors like deep shadows and direct sunlight.

To make my pipeline more robust, I could add training data with the problem conditions and/or find different parameters or other image features that might do better in different conditions.
