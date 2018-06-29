#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 17:37:40 2018

@author: Tyler Wilson and Lydia Masri
"""

import numpy as np
import cv2
import os


image = cv2.imread('/Users/lydiamasri/Desktop/test_card.jpg')
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#card_deck = '/Users/lydiamasri/Desktop/training deck'
test_deck = '/Users/lydiamasri/Desktop/small_deck'


def read_in_images(folder):
    images = []

    for filename in os.listdir(folder):
        # Mac hack
        if filename[0] == '.':
            print('Skipping', filename)
            continue
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def version_zero(image, folder):
        # SIFT settings
    nFeatures = 0
    nOctaveLayers = 5
    contrastThreshold = .1  # Threshold to filter out weak features
    edgeThreshold = 15  # Threshold to filter out edges (lower is stricter)
    sigma = 1.5 # The gaussian std dev at octave zero
    # Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create(nFeatures, nOctaveLayers, contrastThreshold,
                                       edgeThreshold, sigma)
    # Detect keypoints and compute their descriptors
    kp1, des1 = sift.detectAndCompute(gray_img, None)
    img_list = read_in_images(test_deck)
    matches = []
    rows, cols = image.shape[:2]
    output_img = np.zeros((rows, cols, 3), dtype= 'uint8')  
    '''To loop through pieces and find the Brute Force Matches based on the
        number of keypoints found in common between the images.'''
    for img in img_list:   
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(img_g, None)
        h, w = img.shape[:2]
        if des2 is None:
            print('No keypoints found for image 2')
        else:  # Find matches between keypoints in the two images.
            bf = cv2.BFMatcher()
            matches = bf.match(des1, des2)
            if len(matches) == 0:
                print('No matches')
            else:
                print('balls')
                matches = sorted(matches, key=lambda x:x.distance)
                match = matches[0]
                print(match)
#                cv2.imshow("Match", match)
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()
#                curr_kp1 = kp1[match.queryIdx]  # get the keypoint for img1                    
#                loc1 = curr_kp1.pt
#                x1 = int(loc1[0])
#                y1 = int(loc1[1])
#                put_x = (x1//w)*w
#                put_y = (y1//h)*h
#                output_img[put_y:put_y+h, put_x:put_x+w] = img                
#                cv2.imshow('Image 2', output_img)
#                cv2.waitKey(100)    
    return output_img                   
test = version_zero(image, test_deck)
#cv2.imshow("Version 0: Map", test)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
