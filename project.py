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
parser.add_argument("-c", '--calibrate', action='store_true', help="determine camera calibration")
parser.add_argument("-n", '--num_images', type=int, help="number of images to process")
parser.add_argument("-f", '--image_files', type=str, help="file(s) to process, uses glob")
parser.add_argument("-m", '--video', action='store_true', help="process video instead of images")
args = parser.parse_args()

g_debug_internal = False
g_error_frames = 0
g_filename = "none"

# The goals / steps of this project are the following:

# _ 1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
# _ 2. Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
# _ 3. Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
# _ 4. Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
# _ 5. Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# _ 6. Estimate a bounding box for vehicles detected.

def main():

    global g_debug_internal
    if args.verbose:
        print("being verbose")

if __name__ == "__main__":
    main()

