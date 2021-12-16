import cv2
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels = []
file_name = 'Labels.txt'
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

#cap = cv2.VideoCapture(0)
import numpy as np
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
OBJECT_FLAG=0

def object_Run():
    global OBJECT_FLAG
    OBJECT_FLAG=0
    
    cap = cv2.VideoCapture(0)

    while True:
        ret,frame = cap.read()
        ClassIndex, confidece,bbox = model.detect(frame,confThreshold=0.55)
        #print(ClassIndex)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #mask = cv2.inRange(gray, np.array([0,0,0]), np.array([0,0,255]))
        #frame = cv2.bitwise_and(frame,frame, mask= mask)
        if (len(ClassIndex)!=0):

            for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidece.flatten(),bbox):

                if (ClassInd<=80):
                    cv2.rectangle(frame,boxes,(255,0,2),2)
                    cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+0,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)



        cv2.imshow('Object Detection Tutorial',frame )

        if cv2.waitKey(2) and OBJECT_FLAG==1 :
            break
    cap.release()
    cv2.destroyAllWindows()
    
def object_stop():
    global OBJECT_FLAG
    OBJECT_FLAG=1