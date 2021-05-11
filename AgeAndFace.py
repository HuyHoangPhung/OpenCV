# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 15:06:12 2021

@author: HuyHoang
"""
import numpy as np
import cv2
import os
#Age detector function
#Function for age detector
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


#Face detector
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

# Part 1: Create fisherRecognizer
print('Recognizing Face Please Be in sufficient Lights...')

# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(width, height) = (130, 100)

# Create a Numpy array from the two lists above
(images, lables) = [np.array(lis) for lis in [images, lables]]

# OpenCV trains a model from the images
# NOTE FOR OpenCV2: remove '.face'
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables)
# Part 2: Use fisherRecognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
###

#Age detector workaround
args = {
    "face": "face_detector",
    "age": "age_detector",
    "confidence": 0.9
}
# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load our serialized age detector model from disk
print("[INFO] loading age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

###Loop with webcam

while True:
    #Read webcam
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_resize = cv2.resize(gray, (130,100))
    prediction = model.predict(face_resize)
    #Init age result
    results = detect_and_predict_age(
    im, faceNet, ageNet, minConf=args["confidence"])
    for r in results:
        text = "{}: {:.2f}%".format(r["age"][0], r["age"][1]*100)
        (startX, startY, endX, endY) = r["loc"]
        cv2.putText(im, text, (startX-30, startY-30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))   
        #condition
    if prediction[1] < 120:
            cv2.putText(im, '%s - %.0f'%(names[prediction[0]], prediction[1]) , (startX-10, startY-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
    else:
            cv2.putText(im, 'Not recognize', (startX-10, startY-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255,0))      
    cv2.imshow("winname", im)
    key = cv2.waitKey(1) & 0xFF

# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        webcam.release()
        break
    
webcam.release()
cv2.destroyAllWindows()
