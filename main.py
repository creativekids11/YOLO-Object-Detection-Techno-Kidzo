import cv2 as cv
import numpy as np
 
cap = cv.VideoCapture(0)
whT = 320
confThreshold =0.5
nmsThreshold= 0.2
 
#### LOAD MODEL
## Coco Names
classesFile = "coco.names"
classNames = &#91;]
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('n').split('n')
print(classNames)
## Model Files
modelConfiguration = "yolov3-320.cfg"
modelWeights = "yolov3-320.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
 
def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = &#91;]
    classIds = &#91;]
    confs = &#91;]
    for output in outputs:
        for det in output:
            scores = det&#91;5:]
            classId = np.argmax(scores)
            confidence = scores&#91;classId]
            if confidence > confThreshold:
                w,h = int(det&#91;2]*wT) , int(det&#91;3]*hT)
                x,y = int((det&#91;0]*wT)-w/2) , int((det&#91;1]*hT)-h/2)
                bbox.append(&#91;x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
 
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
 
    for i in indices:
        i = i&#91;0]
        box = bbox&#91;i]
        x, y, w, h = box&#91;0], box&#91;1], box&#91;2], box&#91;3]
        # print(x,y,w,h)
        cv.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
        cv.putText(img,f'{classNames&#91;classIds&#91;i]].upper()} {int(confs&#91;i]*100)}%',
                  (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
 
while True:
    success, img = cap.read()
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), &#91;0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = &#91;(layersNames&#91;i&#91;0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs,img)
 
    cv.imshow('Image', img)
    cv.waitKey(1)
