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

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

distortion_coeffs_pickle_file = "camera_cal/dist_pickle.p"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", '--verbose', action='store_true', help="be verbose")
parser.add_argument("-t", '--train', action='store_true', help="setup & train classifier")
#parser.add_argument("-n", '--num_images', type=int, help="number of images to process")
#parser.add_argument("-f", '--image_files', type=str, help="file(s) to process, uses glob")
#parser.add_argument("-m", '--video', action='store_true', help="process video instead of images")
args = parser.parse_args()

g_error_frames = 0
g_filename = "none"

# The goals / steps of this project are the following:

# _ 1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
# _ 2. Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
# _ 3. Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
# _ 4. Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
# _ 5. Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# _ 6. Estimate a bounding box for vehicles detected.

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

    def next(self, image, title=None, just_plot=False):
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
            plt.imshow(image.squeeze(), cmap='gray')

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
                                  block_norm= 'L2-Hys',
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
                       block_norm= 'L2-Hys',
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

def setup_and_train_classifier():

    image_files_vehicle = glob.glob("vehicles/**/*.png")
    image_files_non_vehicle = glob.glob("non-vehicles/**/*.png")

    if False: # for writeup
        plot_vehicle_images(image_files_vehicle, image_files_non_vehicle)
        return

    print("Found {:d} vehicle images, {:d} non-vehicle images".format(len(image_files_vehicle), len(image_files_non_vehicle)))

    #colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    #orient = 11
    #pix_per_cell = 16
    #cell_per_block = 2
    #hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

    # Reduce the sample size for testing because HOG features are slow to compute
    sample_size = 0
    if sample_size == 0:
        cars = image_files_vehicle
        notcars = image_files_non_vehicle
    else:
        print("WARNING: using sample size:", sample_size)
        cars = image_files_vehicle[0:sample_size]
        notcars = image_files_non_vehicle[0:sample_size]

    colorspaces = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
    orientations = [6, 7, 8, 9, 10, 11, 12]
    pixels_per_cell = [8, 16, 32]
    cells_per_block = [2]
    hog_channels = ["ALL"]

    for colorspace in colorspaces:
        for orientation in orientations:
            for pix_per_cell in pixels_per_cell:
                for cell_per_block in cells_per_block:
                    for hog_channel in hog_channels:
                        time_extract_hog, time_train, accuracy = one_hog(cars, notcars, colorspace, orientation, pix_per_cell, cell_per_block, hog_channel)
                        print("color:" + colorspace,
                              "orien:" + str(orientation),
                              "pixel:" + str(pix_per_cell),
                              "cells:" + str(cell_per_block),
                              "chans:" + str(hog_channel),
                              "t_hog:" + str(round(time_extract_hog, 2)),
                              "t_trn:" + str(round(time_train, 2)),
                              "accur:" + str(accuracy))

def one_hog(cars, notcars, colorspace, orient, pix_per_cell, cell_per_block, hog_channel):
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
    return time_extract_hog, time_train, accuracy

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


def main():

    if args.verbose:
        print("being verbose")

    if args.train:
        setup_and_train_classifier()


if __name__ == "__main__":
    main()

