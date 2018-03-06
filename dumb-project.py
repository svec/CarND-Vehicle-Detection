#!/usr/bin/env python3

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os
import math
import pickle
import pdb
import sys
import time

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", '--verbose', action='store_true', help="be verbose")
parser.add_argument("-t", '--train', action='store_true', help="setup & train classifier")
parser.add_argument("-n", '--num_images', type=int, help="number of images to process")
parser.add_argument("-f", '--image_files', type=str, help="file(s) to process, uses glob so you should use '' for expansion")
#parser.add_argument("-m", '--video', action='store_true', help="process video instead of images")
args = parser.parse_args()

g_error_frames = 0
g_filename = "none"

X_train = None
X_test = None
y_train = None
y_test = None
hog_params = {}

# The goals / steps of this project are the following:

# X 1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
# _ 2. Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
# _ 3. Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
# X 4. Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
# _ 5. Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# _ 6. Estimate a bounding box for vehicles detected.

# Here is your draw_boxes function from the previous exercise
# Based on the  Udacity lesson
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

class Subplotter:
    # Handle subplots intelligently.
    def __init__(self):
        self.cols = 0
        self.rows = 0
        self.current = 0

    def setup(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.current = 1
        # Plot the result
        f, ax = plt.subplots(self.rows, self.cols, figsize=(14, 8))
        f.tight_layout()

    def next(self, image, title=None, just_plot=False, cmap='gray'):
        if self.current == 0:
            print("ERROR: subplot next called before setup")
            sys.exit(1)

        if self.current > (self.cols * self.rows):
            print("ERROR: too many subplots for rows, cols:", self.rows, self.cols)
            sys.exit(1)

        plt.subplot(self.rows, self.cols, self.current)

        if just_plot:
            plt.plot(image)
        else:
            plt.imshow(image.squeeze(), cmap=cmap)

        if title:
            plt.title(title)
        self.current = self.current + 1

    def show(self):
        #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

g_subplotter = Subplotter()

def imshow_full_size(img, title=False, *args, **kwargs):
    dpi = 100
    margin = 0.05 # (5% of the width/height of the figure...)
    ypixels, xpixels = img.shape[0], img.shape[1]
    
    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * xpixels / dpi, (1 + margin) * ypixels / dpi
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    ax.imshow(img, interpolation='none', *args, **kwargs)
    if title:
        plt.title(title)
    plt.show()

def plot_images(images_files, title=""):

    for image_file in images_files:
        image = mpimg.imread(image_file)
        g_subplotter.next(image, title)

def plot_hogs(image_files, num_to_plot):

    if num_to_plot > len(image_files):
        print("ERROR: not enough elements in image_files")
        sys.exit(1)

    for ii in range(num_to_plot):
        image = mpimg.imread(image_files[ii])

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Call our function with vis=True to see an image output
        features, hog_image = get_hog_features(gray, orient= 9, 
                                               pix_per_cell= 8, cell_per_block= 2, 
                                               vis=True, feature_vec=False)

        g_subplotter.next(image, "orig")
        g_subplotter.next(hog_image, "hog")

def plot_vehicle_images(image_files_vehicle, image_files_non_vehicles):

    vehicles = image_files_vehicle[0:1000:100]
    non_vehicles = image_files_non_vehicles[0:1000:100]

    g_subplotter.setup(cols=5, rows=4)
    plot_images(vehicles, "vehicles")
    plot_images(non_vehicles, "non-vehicle")
    g_subplotter.show()

    g_subplotter.setup(cols=4, rows=4)
    plot_hogs(vehicles, 4)
    plot_hogs(non_vehicles, 4)
    g_subplotter.show()

# Define a function to return HOG features and visualization
# Based on the Udacity lesson.
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, 
                                  orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  #block_norm= 'L2-Hys',
                                  transform_sqrt=False,  # = True was causes 'nan' features
                                  visualise=vis,
                                  feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, 
                       orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       #block_norm= 'L2-Hys',
                       transform_sqrt=False,  # = True was causes 'nan' features
                       visualise=vis, 
                       feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
# Based on the Udacity lesson.
def extract_features(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features

def setup_training_data():
    # Training data is from video streams, so if we randomly shuffle all images to get training/validation data
    # then the training and validation data will get almost the same images in them.
    # Instead I'll manually separate 20% of each type of view for validation, and then shuffle the resulting
    # training and validation data.
    #   GTI_Far/ 834 files, 20% = 166, image0786.png - image0974.png (166 files)
    #   GTI_Left/ 909 files, 20% = 181, image0781.png - image0974.png (179 files)
    #   GTI_MiddleClose/  419 files, 20% = 84, image0400.png - image0494.png (83 files)
    #   GTI_Right/ 664 files, 20% = 133, image0803.png  - image0160.png (132 files)
    #   KITTI_extracted/ - these are all different enough, no manual sorting is necessary
    image_files_vehicle = glob.glob("vehicles/**/*.png")
    image_files_non_vehicle = glob.glob("non-vehicles/**/*.png")

    if False: # for writeup
        plot_vehicle_images(image_files_vehicle, image_files_non_vehicle)
        return

    print("Found {:d} vehicle images, {:d} non-vehicle images".format(len(image_files_vehicle), len(image_files_non_vehicle)))

    # Reduce the sample size for testing because HOG features are slow to compute
    sample_size = 0
    if sample_size == 0:
        cars = image_files_vehicle
        notcars = image_files_non_vehicle
    else:
        print("WARNING: using sample size:", sample_size)
        cars = image_files_vehicle[0:sample_size]
        notcars = image_files_non_vehicle[0:sample_size]

    return cars, notcars

def setup_and_train_classifier():
    global hog_params

    if args.verbose:
        print("Training classifier (this could take a few minutes)")

    cars, notcars = setup_training_data()

    svc = None

    run_for_real = True

    if run_for_real:
        colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orientation = 10
        pix_per_cell = 16
        cell_per_block = 2
        hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

        hog_params["colorspace"] = colorspace
        hog_params["orientation"] = orientation
        hog_params["pix_per_cell"] = pix_per_cell
        hog_params["cell_per_block"] = cell_per_block
        hog_params["hog_channel"] = hog_channel

        time_extract_hog, time_train, accuracy, svc = one_hog(cars, notcars, colorspace, orientation, pix_per_cell, cell_per_block, hog_channel)

        print("color:" + colorspace,
              "orien:" + str(orientation),
              "pixel:" + str(pix_per_cell),
              "cells:" + str(cell_per_block),
              "chans:" + str(hog_channel),
              "t_hog:" + str(round(time_extract_hog, 2)),
              "t_trn:" + str(round(time_train, 2)),
              "accur:" + str(accuracy))
        sys.stdout.flush()

        test_trained_svc(svc)

    else:
        sys.stdout.flush()

        # Training runs
        colorspaces = ['LUV', 'YUV', 'RGB', 'HSV', 'HLS', 'YCrCb']
        orientations = [6, 7, 8, 9, 10, 11, 12]
        pixels_per_cell = [8, 16, 32]
        cells_per_block = [2]
        hog_channels = ["ALL"]
        
        # Best run from these:
        # color	orientation	pixels	cells	channels	time hog	time train	total time	accuracy
        # LUV	9	16	2	ALL	48.7	4.74	53.44	0.9817
        for colorspace in colorspaces:
            for orientation in orientations:
                for pix_per_cell in pixels_per_cell:
                    for cell_per_block in cells_per_block:
                        for hog_channel in hog_channels:
                            time_extract_hog, time_train, accuracy, svc = one_hog(cars, notcars, colorspace, orientation, pix_per_cell, cell_per_block, hog_channel)
                            print("color:" + colorspace,
                                  "orien:" + str(orientation),
                                  "pixel:" + str(pix_per_cell),
                                  "cells:" + str(cell_per_block),
                                  "chans:" + str(hog_channel),
                                  "t_hog:" + str(round(time_extract_hog, 2)),
                                  "t_trn:" + str(round(time_train, 2)),
                                  "accur:" + str(accuracy))
                            sys.stdout.flush()
    return svc

def test_trained_svc(trained_svc):
    global X_test
    global y_test

    accuracy = round(trained_svc.score(X_test, y_test), 4)
    print("test accuracy:", accuracy)

def one_hog(cars, notcars, colorspace, orient, pix_per_cell, cell_per_block, hog_channel):
    global X_train
    global X_test
    global y_train
    global y_test
    t=time.time()
    car_features = extract_features(cars, cspace=colorspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
    notcar_features = extract_features(notcars, cspace=colorspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
    t2 = time.time()
    time_extract_hog = t2-t
    #print(round(t2-t, 2), 'Seconds to extract HOG features...')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    if False: # Only needed if combining features other than HOG
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)
    
    #print('Using:',orient,'orientations',pix_per_cell,
        #'pixels per cell and', cell_per_block,'cells per block')
    #print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    time_train = t2-t
    #print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    accuracy = round(svc.score(X_test, y_test), 4)
    return time_extract_hog, time_train, accuracy, svc

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print(X_test[0:n_predict].shape)
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    return time_extract_hog, time_train, accuracy, svc

# Define a single function that can extract features using hog sub-sampling and make predictions
# Based on the Udacity lesson.
def find_cars(img, ystart, ystop, scale, svc, X_scaler, colorspace, orientation, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, verbose_image, verbose_color=(0,0,255)):

    show_all_boxes = False

    if args.verbose:
        show_all_boxes = True
        #print("y: {:d}-{:d} = {:d} pixels, scale: {:f}".format(ystart, ystop, ystop-ystart+1, scale))
        #cv2.rectangle(verbose_image, (0,ystart), (img.shape[1],ystop), verbose_color, 6) 

    rectangles_found = []

    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]

    if colorspace != 'RGB':
        if colorspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif colorspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif colorspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif colorspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif colorspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(image)      

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    #if args.verbose:
        #plt.imshow(ctrans_tosearch)
        #plt.show()

    if hog_channel == "ALL":
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    else:
        ch1 = ctrans_tosearch[:,:,hog_channel]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orientation*cell_per_block**2
    #print("nxblocks =", nxblocks, "= x_width  {:d} // pix_per_cell {:d} - cell_per_block {:d} + 1".format( \
    #      ch1.shape[1], pix_per_cell, cell_per_block))
    #print("nyblocks =", nyblocks, "= y_height {:d} // pix_per_cell {:d} - cell_per_block {:d} + 1".format( \
    #      ch1.shape[0], pix_per_cell, cell_per_block))
    #print("nfeat_per_block =", nfeat_per_block, "= orientation {:d} * cell_per_block {:d} ** 2".format( \
    #      orientation, cell_per_block))
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    #print("nxsteps, nysteps:", nxsteps, nysteps)
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orientation, pix_per_cell, cell_per_block, feature_vec=False)
    if hog_channel == "ALL":
        hog2 = get_hog_features(ch2, orientation, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orientation, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            #print("step x {:2d} y {:2d} : x,y: {:3d}, {:3d}".format(xb, yb, xpos, ypos))
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            if hog_channel == "ALL":
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            if False: # only using hog features
                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = svc.predict(test_features)

            # Reshape the hog features to be a 1-sample 2D array (required for svc.predict()).
            # (Note, could have also just done hog_features = [hog_features]
            hog_features = hog_features.reshape(1,-1)
            test_prediction = svc.predict(hog_features)
            
            if (test_prediction == 1) or show_all_boxes:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                upper_left = (xbox_left, ytop_draw+ystart)
                lower_right = (xbox_left+win_draw,ytop_draw+win_draw+ystart)
                #print("  rect:", upper_left, lower_right)
                if show_all_boxes:
                    temp_color = verbose_color
                    temp_width = 2
                    if test_prediction == 1:
                        temp_color = (255,0,0)
                        temp_width = 6
                    cv2.rectangle(verbose_image, upper_left, lower_right, temp_color, temp_width) 

                if test_prediction == 1:
                    rectangles_found.append((upper_left, lower_right))

    if args.verbose:
        print("Found", len(rectangles_found), "boxes")
        #imshow_full_size(verbose_image)
                
    #return draw_img
    return rectangles_found

# Based on the Udacity lesson.
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

# Based on the Udacity lesson.
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

# Based on the Udacity lesson.
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def process_one_image(image, svc):
    global hog_params
    X_scaler = None
    spatial_size = None
    hist_bins = None

    verbose_image = np.copy(image)

    box_list = []

    find_car_runs = [  # ystart, ystop, scale, verbose_color
                       [400, 464, 1.0, (0,0,255)],
                       [416, 480, 1.0, (0,0,192)],
                       [432, 496, 1.0, (0,0,128)],
                       [464, 528, 1.0, (0,0,64)],

                       #[400, 464, 1.0, (0,0,255)],

                       [400, 528, 2.0, (255,255,0)],
                       [528, 656, 2.0, (0,255,255)],
                    ]
    for run in find_car_runs:
        ystart = run[0]
        ystop = run[1]
        scale = run[2]
        verbose_color = run[3]
        box_list += find_cars(image, ystart, ystop, scale, svc, X_scaler,
                                 hog_params["colorspace"],
                                 hog_params["orientation"],
                                 hog_params["pix_per_cell"],
                                 hog_params["cell_per_block"],
                                 hog_params["hog_channel"],
                                 spatial_size,
                                 hist_bins,
                                 verbose_image,
                                 verbose_color = verbose_color)

    out_img = draw_boxes(image, box_list, color=(255,0,0),thick=6)

    #imshow_full_size(out_img)

    # Add heat to each box in box list
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat,box_list)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    g_subplotter.setup(cols=2, rows=2)
    g_subplotter.next(verbose_image)
    g_subplotter.next(out_img)
    g_subplotter.next(draw_img)
    g_subplotter.next(heatmap, cmap='hot')
    g_subplotter.show()

def process_image_file(image_filename, svc):
    global g_filename
    g_filename = image_filename

    if args.verbose:
        print(image_filename)

    image = mpimg.imread(image_filename) # reads in as RGB
    process_one_image(image, svc)

def process_images(svc, num, filenames=None):
    if filenames:
        image_filenames = glob.glob(filenames)
        #image_filenames = image_filenames + glob.glob('test_images/*.jpg')
    else:
        image_filenames = glob.glob('test_images/*.jpg')

    #print("files:", image_filenames)
    count = 0
    for image_filename in image_filenames:
        if (num != None) and (count >= num):
            break
        process_image_file(image_filename, svc)
        count = count + 1

def main():
    global X_train
    global X_test
    global y_train
    global y_test
    global hog_params

    if args.verbose:
        print("being verbose")

    pickle_filename = "dumb-trained-svc.p"
    svc = None

    if args.train:
        svc = setup_and_train_classifier()
        svc_pickle = {}
        svc_pickle["svc"] = svc
        svc_pickle["X_train"] = X_train
        svc_pickle["X_test"] = X_test
        svc_pickle["y_train"] = y_train
        svc_pickle["y_test"] = y_test
        svc_pickle["hog_params"] = hog_params
        pickle.dump(svc_pickle, open(pickle_filename, "wb"))
        print("Saved trained SVM data in", pickle_filename)
    else:
        svc_pickle = pickle.load( open( pickle_filename, "rb" ) )
        svc = svc_pickle["svc"]
        X_train = svc_pickle["X_train"]
        X_test = svc_pickle["X_test"]
        y_train = svc_pickle["y_train"]
        y_test = svc_pickle["y_test"]
        hog_params = svc_pickle["hog_params"]
        print("Read trained SVM data from", pickle_filename)
        print("hog_params:", hog_params)
        test_trained_svc(svc)

        process_images(svc, args.num_images, args.image_files)

if __name__ == "__main__":
    main()

