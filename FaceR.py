
# # Useful

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import threading
# from PIL import ImageGrab
 
path = 'ImagesBasic'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('Encoding Complete')


import time
cord = []
pord = []

def face_recog_start():
    global FACE_FLAG,p
    FACE_FLAG=0
    cap = cv2.VideoCapture(0)
    ##################################
    ##################################
    def image_display(taskqueue,lask):
        #print("hell")
        global cord , pord
       #cv2.namedWindow ('image_display', cv2.CV_WINDOW_AUTOSIZE)
        piy = 0
        start_time = time.perf_counter()

        while True:
            if cv2.waitKey(2) and FACE_FLAG==1 :
                break
            #print("hell0")
            image = taskqueue.get()
            facesCurFrame = lask.get()

            if piy==0 or len(pord)==0 or (time.perf_counter() - start_time)> 5.0:
                print("Works",(time.perf_counter() - start_time))
                encodesCurFrame =face_recognition.face_encodings(image,facesCurFrame)
                start_time = time.perf_counter()
                pord.clear()
                piy=0
            #print(encodesCurFrame)
            #print(facesCurFrame)
            cord.clear()
            cord.append(facesCurFrame)
            pord.append(encodesCurFrame)
            piy = 1


    from multiprocessing import Process
    from queue import Queue
    import threading
    import time
    taskqueue = Queue(maxsize = 3)
    lask = Queue()
    p = threading.Thread(target=image_display, args=(taskqueue,lask))
    p.start()

    count=0
    while True:

        success, img = cap.read()
        #img = captureScreen()
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        taskqueue.put(imgS)
        facesCurFrame = face_recognition.face_locations(imgS)
        if len(facesCurFrame)>0:
            facesCurFrame=[facesCurFrame[0]] 
            print(face_recognition.face_locations(imgS))
        lask.put(facesCurFrame)
        #print(taskqueue)
        #facesCurFrame = face_recognition.face_locations(imgS)
        #encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
        #print(facesCurFrame)
        #print(encodesCurFrame)
        if count >=3 :
            #print(count)

            for encodeFace,faceLoc in zip(pord[0],cord[0]):#zip(encodesCurFrame,facesCurFrame):
                matches =face_recognition.compare_faces(encodeListKnown,encodeFace)
                faceDis =face_recognition.face_distance(encodeListKnown,encodeFace)
                #print(matches)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    #print(name)
                    y1,x2,y2,x1 = faceLoc
                    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                    #markAttendance(name)
            cv2.imshow('Webcam',img)
            cv2.waitKey(1)

        if count < 3:
            time.sleep(2)
            count=count+1
            pass
        
        if cv2.waitKey(2) and FACE_FLAG==1 :
            break
    cap.release()
    cv2.destroyAllWindows()
    p.join()


def face_recog_stop():
    global FACE_FLAG,p
    FACE_FLAG=1
    while True:
        if p.is_alive():
            time.sleep(2)
        else:
            break
        

