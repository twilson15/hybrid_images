#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:12:56 2018

@author: Tyler Wilson
"""
import cv2
import numpy as np
import os

#example_image_path = '../data/positive/train/1.jpg'

'''This is the function I use to read in each image or file from the folder
    and return a list. Used for positive/negative training/testing''' 
def read_in_pieces(folder):
#    j = 0
    img_pieces = []
    x_coords = []
    y_coords = []
    eye_cascade = cv2.CascadeClassifier('/Users/tylerwilson/Desktop/ComputerVision/hw05/haarcascade_eye.xml')
    os.chdir(folder)
    for f in os.listdir(folder):
        f_img = cv2.imread(f)
        img_pieces.append(f_img)
#        eye_cascade = cv2.CascadeClassifier('/Users/tylerwilson/Desktop/ComputerVision/hw05/haarcascade_eye.xml')
        eye_detect = eye_cascade.detectMultiScale(f_img, 1.3, 5)
        for (x, y, w, h) in eye_detect:      
            x_coords.append(x)
            y_coords.append(y)
#            disp = f.copy()
#            disp = cv2.rectangle(disp, (x, y), (x+w, y+h), (100, 200, 0), 2)
#            if j == 0:
#                eye1 = disp[y:y+h,x:x+w]
#                cv2.imshow("face 1", eye1)
#                cv2.waitKey(0)
#                j+=1
#            if j == 1:
#                eye2 = disp[y:y+h,x:x+w]
#                cv2.imshow("face 2", eye2)
#                cv2.waitKey(0)
#                j+=1     
#        print(x_coords)
#        print(y_coords)           
#            face_orient = np.arctan((y2-y1)/(x2-x1))
#            face_image = cv2.resize(face_image, (150, 150))  
    return img_pieces


'''Reads in either all the training images or the testing images and adds 
    the negatives and positive images to one list. It also adds their binary 
    equivalent, 1 for positive or 0 for negative, to a list.'''
def read_test_train(positive, negative):
    label_list_training = []
    face_images = []
    for pos in positive:
        pos = cv2.cvtColor(pos, cv2.COLOR_BGR2GRAY)
        face_images.append(pos)
        label_list_training.append(1)
#    return label_list_training
      
    for neg in negative: 
        neg = cv2.cvtColor(neg, cv2.COLOR_BGR2GRAY)
        face_images.append(neg)
        label_list_training.append(0)
    return face_images, label_list_training


''' Training ''' 

train_positives = read_in_pieces('/Users/tylerwilson/Desktop/ComputerVision/hw05/positive/train')
train_negatives = read_in_pieces('/Users/tylerwilson/Desktop/ComputerVision/hw05/negative/train')
'''Make lists for training''' 
train_images, label_list_training = read_test_train(train_positives, train_negatives)
labels_train = np.array(label_list_training)
'''Create Eigenface detector'''
model = cv2.face.EigenFaceRecognizer_create()
model.train(train_images, labels_train)

'''Get the mean of the model to see if it's on right track...
    Shows the 3 first Eigenfaces of the model to see principle features''' 
#Show the average of model to see if things look right
m = model.getMean()
m = np.reshape(m, (150, 150))
m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imshow('Mean Face', m)
cv2.waitKey(0)
cv2.destroyAllWindows()
#To show first 3 Eigenfaces photos of the model
eigen_vecs = model.getEigenVectors()
'''Loop to show Eigenfaces''' 
for i in range(3):
    f = np.reshape(eigen_vecs[:, i], (150, 150))
    f = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow('Eigenface', f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

''' Testing '''
'''Get positive and negative test folders''' 
pos_test = read_in_pieces('/Users/tylerwilson/Desktop/ComputerVision/hw05/positive/test')
neg_test = read_in_pieces('/Users/tylerwilson/Desktop/ComputerVision/hw05/negative/test')

'''This model calls the read_train_test function to get the two lists for the 
    testing set. Keeps track of tp, tn, fp, and fn. Uses those numbers to 
    calculate precision, recall, and accuracy of model.''' 
def test_model(positives, negatives, live_debug):
    pred_list = []
    test_images, test_label_list = read_test_train(positives, negatives)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    t = 0
    for test in test_images:
    #    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        pred_label, confidence = model.predict(test)
        pred_list.append(pred_label)   
    #    print(pred_label)
    #    print(test_label_list)
        if pred_label == 1 and test_label_list[t] == 1:
    #        print (pred_label)
            tp += 1
        if pred_label == 0 and test_label_list[t] == 0:
            tn += 1
        if pred_label == 1 and test_label_list[t] == 0:
            fp += 1
        if pred_label == 0 and test_label_list[t] == 1:
            fn += 1   
        t+=1   
    accuracy = (tp + tn) / (tp + tn + fp + fn) 
    recall =  tp / (tp + fn)
    precision = tp / (tp + fp) 
    print('True positives: {}'.format(tp))
    print('True negatives: {}'.format(tn))
    print('False positives: {}'.format(fp))
    print('False negatives: {}'.format(fn))
    print('Accuracy = {:.2f}%'.format(accuracy))
    print('Recal: {:.2f}%'.format(recall))
    print('Precision: {:.2f}%'.format(precision))
    
    '''This shows a live video demo of the model working to detect which faces
        are mine, or positive, and which faces are not mine, meaning negative.
        Only shows demo if param live_demo is True'''
    if live_debug == True:
        i = 1
        face_cascade = cv2.CascadeClassifier('/Users/tylerwilson/Desktop/ComputerVision/hw05/haarcascade_frontalface_default.xml')
        vid = cv2.VideoCapture(0)
        # Loop forever (until user presses q)
        while True:
            ret, frame = vid.read()
        # Check the return value to make sure the frame was read successfully
            if not ret:
                print('Error reading frame')
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            disp = frame.copy()
            # Detect faces in the gray image. (Lower numbers will detect more faces.)
            # Parameters:
            #   scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
            #   minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
            face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
            # loop through detected face rectangles to identify
            for (x, y, w, h) in face_rects:
                # Draw a rectangle around the detected face
                disp = cv2.rectangle(disp, (x, y), (x+w, y+h), (100, 200, 0), 2)
                face_image = np.zeros((h, w, 3), dtype = 'uint8')
                face_image = disp[y:y+h, x:x+h] 
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image = cv2.resize(face_image, (150, 150))
                pred_label2, confidence2 = model.predict(face_image)
                if pred_label2 == 1:
                    disp = cv2.rectangle(disp, (x, y), (x+w, y+h), (100, 200, 0), 2)
                if pred_label2 == 0:
                    disp = cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 0, 200), 2)
            cv2.imshow('Video', disp)
            # Get which key was pressed
            key = cv2.waitKey(1)
            # Keeps looping until the user presses 'q'
            if key & 0xFF == ord('q'):
                break
            # Saves current frame when user presses key 's'
            if 's' == chr(key & 255):
                cv2.imwrite(str(i).zfill(3) + '.jpg', disp)
                int(i)
                i += 1
        vid.release()
        cv2.destroyAllWindows()
test_model(pos_test, neg_test, True)
cv2.destroyAllWindows()

'''This function was to be used when I figured out the offset angle of the eyes'''
#def rotate_image(img, rot_angle):
#    rows, columns = img.shape[:2]
#    img_center = (columns//2,rows//2)
#    M = cv2.getRotationMatrix2D(img_center,rot_angle, 1.0)
#    img_rotated = cv2.warpAffine(img, M, (columns, rows))
#    # return the rotated image
#    return img_rotated


