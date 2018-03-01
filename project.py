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

def plot_vehicle_images(image_files_vehicle, image_files_non_vehicles):

    vehicles = image_files_vehicle[0:1000:100]
    non_vehicles = image_files_non_vehicles[0:1000:100]

    g_subplotter.setup(cols=5, rows=4)
    plot_images(vehicles, "vehicles")
    plot_images(non_vehicles, "non-vehicle")
    g_subplotter.show()

def setup_and_train_classifier():

    image_files_vehicle = glob.glob("vehicles/**/*.png")
    image_files_non_vehicle = glob.glob("non-vehicles/**/*.png")

    if False:
        plot_vehicle_images(image_files_vehicle, image_files_non_vehicle)

    print("Found {:d} vehicle images, {:d} non-vehicle images".format(len(image_files_vehicle), len(image_files_non_vehicle)))

def main():

    if args.verbose:
        print("being verbose")

    if args.train:
        setup_and_train_classifier()


if __name__ == "__main__":
    main()

