# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:38:52 2021

@author: HuyHoang
"""

#library import
import numpy as np
import cv2
import os
#Self create function
def detect_and_predict_age(frame, faceNet, ageNet, minConf=0.7):
    # define the list of age buckets our age detector will predict
    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
                "(38-43)", "(48-53)", "(60-100)"]

    # initialize our results list
    results = []

    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                              (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > minConf:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the ROI of the face
            face = frame[startY:endY, startX:endX]

            # ensure the face ROI is sufficiently large
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue

            # construct a blob from *just* the face ROI
            faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                    (78.4263377603, 87.7689143744, 114.895847746),
                                    swapRB=False)

            # make predictions on the age and find the age bucket with
            # the largest corresponding probability
            ageNet.setInput(faceBlob)
            preds = ageNet.forward()
            i = preds[0].argmax()
            age = AGE_BUCKETS[i]
            ageConfidence = preds[0][i]

            # construct a dictionary consisting of both the face
            # bounding box location along with the age prediction,
            # then update our results list
            d = {
                "loc": (startX, startY, endX, endY),
                "age": (age, ageConfidence)
            }
            results.append(d)

    # return our results to the calling function
    return results