# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 19:39:11 2022

@author: Dell
"""

import cv2
import numpy as np
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
#cap = cv2.VideoCapture('video.mp4')#to take input as veideo
#cap = cv2.VideoCapture(0)#to take input from camera
cap = cv2.VideoCapture('video.mp4')
classes = []
with open('coco.txt', 'r') as f:
    classes = f.read().splitlines()

#cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(260, 3))
while True:
    #cap =  cv2.VideoCapture("http://192.168.50.55:8080//shot.jpg")# to take input from url
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    counterveh = 0
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                if class_id >0 and class_id<8:
                    counterveh +=1
                print("scores:" + str(scores))
                print("class id :"+str(class_id))
                print("confidence:"+str(confidence))
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i] 
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 3)
            cv2.putText(img,label, (x, y+20), font, 2,(130,240,139), 2)
    cv2.putText(img,"no. of vehicle present:-"+str(counterveh),(450,100),font,2,(0,0,255),3)
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()