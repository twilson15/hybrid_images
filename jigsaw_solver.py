#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 11:41:06 2018

@author: Tyler Wilson

I got help with os.listdir() on stackoverflow:
https://stackoverflow.com/questions/46583512/
python-3-6-looping-through-files-in-os-listdir-and-writing-some-of-them-to-a
"""
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

#image = cv2.imread('waterfall.jpg')
#gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image = cv2.imread('hawaii_full.jpg')
#gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image = cv2.imread('hawaii_small.jpg')
#gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.imread('map_small.jpg')
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image = cv2.imread('train_small.jpg')
#gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image = cv2.imread('trains_full.jpg')
#gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image = cv2.imread('map_full.jpg')
#gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#mypath_h_aligned='/Users/tylerwilson/Desktop/ComputerVision/hawaii/pieces_aligned'
#mypath_t_aligned='/Users/tylerwilson/Desktop/ComputerVision/trains/pieces_aligned'
mypath_m_aligned='/Users/tylerwilson/Desktop/ComputerVision/map/pieces_aligned'

'''This function loops through each file in a given folder in order to read in 
    the image and append it to an empty list.'''
def read_in_pieces(folder):
    os.chdir(folder)
    img_pieces = []
    for f in os.listdir():
        f_img = cv2.imread(f)
        img_pieces.append(f_img)
    return img_pieces

def version_zero(image, folder):
        # SIFT settings
    nFeatures = 0
    nOctaveLayers = 5
    contrastThreshold = .001  # Threshold to filter out weak features
    edgeThreshold = 15  # Threshold to filter out edges (lower is stricter)
    sigma = 1 # The gaussian std dev at octave zero
    # Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create(nFeatures, nOctaveLayers, contrastThreshold,
                                       edgeThreshold, sigma)
    # Detect keypoints and compute their descriptors
    kp1, des1 = sift.detectAndCompute(gray_img, None)
    img_list = read_in_pieces(folder)  
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
                matches = sorted(matches, key=lambda x:x.distance)
                match = matches[0]                            
                curr_kp1 = kp1[match.queryIdx]  # get the keypoint for img1                    
                loc1 = curr_kp1.pt
                x1 = int(loc1[0])
                y1 = int(loc1[1])
                put_x = (x1//w)*w
                put_y = (y1//h)*h
                output_img[put_y:put_y+h, put_x:put_x+w] = img                
#                cv2.imshow('Image 2', output_img)
#                cv2.waitKey(100)    
    return output_img                   
#test = version_zero(image, mypath_m_aligned)
#cv2.imshow("Version 0: Map", test)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

def rotate_image(img, rot_angle):
    'This function will rotate the images to see where they align best.'
    rows, columns = img.shape[:2]
    img_center = (columns//2,rows//2)
    M = cv2.getRotationMatrix2D(img_center,rot_angle, 1.0)
    img_rotated = cv2.warpAffine(img, M, (columns, rows))
    # return the rotated image
    return img_rotated

#mypath_m_random='/Users/tylerwilson/Desktop/ComputerVision/map/pieces_aligned'
#mypath_h_random = '/Users/tylerwilson/Desktop/ComputerVision/hawaii/pieces_random'
#mypath_t_random='/Users/tylerwilson/Desktop/ComputerVision/trains/pieces_random'

def version_one(image, folder):
        # SIFT settings
    nFeatures = 0
    nOctaveLayers = 5
    contrastThreshold = .000  # Threshold to filter out weak features
    edgeThreshold = 15  # Threshold to filter out edges (lower is stricter)
    sigma = 1.2 # The gaussian std dev at octave zero
    # Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create(nFeatures, nOctaveLayers, contrastThreshold,
                                       edgeThreshold, sigma)
    # Detect keypoints and compute their descriptors
    kp1, des1 = sift.detectAndCompute(gray_img, None)
    img_list = read_in_pieces(folder)  
    matches = []
    rows, cols = image.shape[:2]
    output_img = np.zeros((rows, cols, 3), dtype= 'uint8')  
    '''To loop through pieces and find the Brute Force Matches based on the
        number of keypoints found in common between the images.'''
    for img in img_list:   
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(img_g, None)
        if des2 is None:
            print('No keypoints found for image 2')
        else:  # Find matches between keypoints in the two images.
            bf = cv2.BFMatcher()
            matches = bf.match(des1, des2)
            if len(matches) == 0:
                print('No matches')
            else:
                matches = sorted(matches, key=lambda x:x.distance)               
                match = matches[0]                            
                curr_kp1 = kp1[match.queryIdx]  # get the keypoint for img1    
                curr_kp2 = kp2[match.trainIdx]
                loc1 = curr_kp1.pt
                x1 = int(loc1[0])
                y1 = int(loc1[1])
                h, w = img.shape[:2]
                put_x = (x1//50)*50
                put_y = (y1//50)*50
                angle1 = curr_kp1.angle
                angle2 = curr_kp2.angle
                angle_change = angle2 - angle1                    
                if angle_change < 0:
                    angle_change += 360                
                if 20 < angle_change < 100:
                    angle_change = 90
                if 100 < angle_change < 190:
                    angle_change= 180
                if 190 < angle_change < 300:
                    angle_change = 270
                fixed_pieces = rotate_image(img, angle_change)   
                output_img[put_y:put_y+50, put_x:put_x+50] = fixed_pieces
#                    cv2.imshow("please work", output_img)
#                    cv2.waitKey(10)
    return output_img                   
#test = version_one(image, mypath_t_random)
#test = version_one(image, mypath_m_random) 
#test = version_one(image, mypath_h_random) 
#cv2.imshow("Verson 1: Train", test)
#cv2.waitKey(0)
#cv2.destroyAllWindows()    
'''This version is intented to take a smaller guide image but produce the same
    output sizze for the puzzle solution. I attempted to find a ratio between 
    size of the small image and the area of all the pieces together in order
    to determine the size of the window and height and width.'''
def version_two(image, folder):
    img_list = read_in_pieces(folder)
        # SIFT settings    
    nFeatures = 0
    nOctaveLayers = 5
    contrastThreshold = .001  # Threshold to filter out weak features
    edgeThreshold = 15  # Threshold to filter out edges (lower is stricter)
    sigma = 1.2 # The gaussian std dev at octave zero
    # Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create(nFeatures, nOctaveLayers, contrastThreshold,
                                       edgeThreshold, sigma)
    # Detect keypoints and compute their descriptors
    kp1, des1 = sift.detectAndCompute(gray_img, None)
    img_list = read_in_pieces(folder)  
    matches = []
    rows, cols = image.shape[:2]
    print(image.shape[:2])
    
    rows2, cols2 = img_list[0].shape[:2]
    large_area = (rows2*cols2) * len(img_list)
    small_area = rows*cols 
    area_ratio = large_area/small_area
    resize_height = int(area_ratio * rows)
    resize_width = int(area_ratio * cols)
   
    output_img = np.zeros((resize_height,resize_width, 3), dtype= 'uint8')
    print(output_img.shape[:2])
    
    '''To loop through pieces and find the Brute Force Matches based on the
        number of keypoints found in common between the images.'''
    for img in img_list:   
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(img_g, None)
        if des2 is None:
            print('No keypoints found for image 2')
        else:  # Find matches between keypoints in the two images.
            bf = cv2.BFMatcher()
            matches = bf.match(des1, des2)
            if len(matches) == 0:
                print('No matches')
            else:
                matches = sorted(matches, key=lambda x:x.distance)
                num_matches_to_show = 1
            # Loop through the top matches
                for i in range(num_matches_to_show):
                    match = matches[i]                            
                    curr_kp1 = kp1[match.queryIdx]  # get the keypoint for img1    
                    curr_kp2 = kp2[match.trainIdx]
                    loc1 = curr_kp1.pt 
                    x1 = loc1[0] * area_ratio
                    y1 = loc1[1] * area_ratio
#                    x1 = x1*area_ratio
#                    y1 = y1*area_ratio
                    put_x = (int(x1)//50)*50
                    print(put_x)
                    put_y = (int(y1)//50)*50
                    print(put_y)
                    angle1 = curr_kp1.angle
                    angle2 = curr_kp2.angle
                    angle_change = angle2 - angle1                    
                    if angle_change < 0:
                        angle_change += 360                
                    if 20 < angle_change < 100:
                        angle_change = 90
                    if 100 < angle_change < 190:
                        angle_change= 180
                    if 190 < angle_change < 300:
                        angle_change = 270
                    fixed_pieces = rotate_image(img, angle_change)            
                    output_img[put_y:put_y+rows2, put_x:put_x+cols2] = fixed_pieces
#                               
                    cv2.imshow("test", output_img)
                    cv2.waitKey(20)   
pleasework = version_two(image, mypath_m_aligned)                    
cv2.imshow("test", pleasework)
cv2.waitKey(0)
cv2.destroyAllWindows()      


def image_to_pieces(image):
    rows, cols = image.shape[:2]
    