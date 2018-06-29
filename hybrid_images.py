#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 19:39:48 2018

@author: twilson15
"""

import cv2
import numpy as np

def make_gaussian_pyramid(image, levels, debug):
    
#    creates and blurs the initial image
    img = cv2.imread(image)
    rows, cols = img.shape[:2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian_pyramid = []
    gaussian_pyramid.append(img_gray)
    counter = 0
   
#    blurs and resizes requested number of times
    while counter < levels - 1:
        
       img_gray = cv2.GaussianBlur(img_gray, (9,9), 0)
       img_gray = cv2.resize(img_gray, (0,0), fx = .5, fy = .5)
       gaussian_pyramid.append(img_gray)
       counter += 1
    
    if debug is True:
        
        for i in gaussian_pyramid:
            cv2.imshow('gaussian', i)
            cv2.waitKey(1000)
        
    return gaussian_pyramid



def make_laplacian_pyramid(image, levels, debug):
    img = cv2.imread(image)
    rows, cols = img.shape[:2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_pyramid = []
    counter = 0
    
    while counter < levels:
        
        blurred_img = (cv2.GaussianBlur(img_gray, (9,9), 0)).astype(float)
        laplacian_img = (img_gray - blurred_img)
        laplacian_pyramid.append(laplacian_img)
        blurred_img = cv2.resize(blurred_img, (0,0), fx = .5, fy = .5)
        img_gray = blurred_img
        counter += 1
        
    if debug is True:
        
        for i in laplacian_pyramid:
            i = cv2.normalize(i, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imshow('laplacian', i)
            cv2.waitKey(1000)

    return laplacian_pyramid

def show_pyramid(pyramid):
    
#    creates initial array and places first image inside of it
    rows, cols = pyramid[0].shape[:2]
    display_cols = int(cols * 1.5)
    display = np.zeros((rows, display_cols), np.uint8)
    current_col = 0
    current_row = 0
    first_image = cv2.normalize(pyramid[0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    display[current_row:rows, current_col:cols] = first_image 
    current_col += cols
    counter = 1
    
#    moves to the right of the initial image and places images of smaller
#    size downward until levels request is met
    while counter < len(pyramid):
        rows, cols = pyramid[counter].shape[:2]
        image_norm = cv2.normalize(pyramid[counter], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        display[current_row:current_row + rows, current_col:current_col + cols] = image_norm
        current_row = current_row + rows
        counter += 1
        
    cv2.imshow('display', display)
    cv2.waitKey(0)
        
        

def make_hybrid_image(img1, img2, levels, split):
    
#    reads in both images and makes the necessary pyramids
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)
    rows1, cols1 = image1.shape[:2]
    rows2, cols2 = image2.shape[:2]
    gaussian_pyramid2 = make_gaussian_pyramid(img2, levels, False)
    laplacian_pyramid1 = make_laplacian_pyramid(img1, levels, False)
    counter = 1
    img1_counter = 1
    img2_counter = 0
    
#    sets base image to begin creating hybrid image with
    hybrid_image = laplacian_pyramid1[0]
    
    while counter <= levels:
        
#          puts the requested number of laplacian images into the hybrid
         if counter <= split:
             laplacian_pyramid1[img1_counter] = cv2.resize(laplacian_pyramid1[img1_counter], (cols1, rows1))
             hybrid_image += laplacian_pyramid1[img1_counter]
             img1_counter += 1
             counter += 1
             
#          puts the requested number of gaussian images into the hybrid
         else:
            gaussian_pyramid2[img2_counter] = cv2.resize(gaussian_pyramid2[img2_counter], (cols1, rows1))
            hybrid_image += gaussian_pyramid2[img2_counter]
            img2_counter += 1
            counter += 1

#   normalized and returns the image
    hybrid_image = cv2.normalize(hybrid_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return hybrid_image


    
image1_name = 'trendycow.jpg'
image2_name = 'trendysheep.jpg'
display = make_hybrid_image(image1_name, image2_name, 7, 2)
first = make_gaussian_pyramid(image2_name, 7, False)
second = make_laplacian_pyramid(image1_name, 7, False)
show_pyramid(first)
show_pyramid(second)
cv2.imshow('mandela and freeman', display)
cv2.waitKey(0)
cv2.destroyAllWindows()

    
    
    