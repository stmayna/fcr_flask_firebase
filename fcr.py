import cv2
import numpy as np
import tensorflow as tf
import mysql_connector as sq
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from flask import render_template
from datetime import datetime
from statistics import mode
from firestore import markAttendanceIntoCloud

# Load model
new_model = tf.keras.models.load_model('../../ProjekTAGoogleColab/model_inference_for_16_faces_model12_RMSprop.h5')
face_cascade = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')

# Settings
namaLabel = ['alam suminto','alvira sudirman', 'fadillah hafis', 'gede ananda adi apriliawan', 'gusti ngurah bagus amarry krisna', 'i kadek dwi gita purnama pramudya', 'i made dwiki satria wibawa', 'i made parasya maharta', 'i putu augi oka adiana', 'i putu kaesa wahyu prana aditya', 'kurniawan sudirman', 'muhammad ilham maulana', 'namira aulia', 'putu irianti putri astari', 'putu kerta adi pande', 'rassya hilabih']
size = (224, 224)
namaObjek= []

# Defining a function that will do the detections
def detect(gray, frame, namaObjek):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    roi_color=[]
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, namaObjek, (x,y+h+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255))
        #cv2.putText(frame,str(score),(x,y-2),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    return frame

# To crop the frame
def getFrameCrop(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = frame[y:y+h, x:x+w]
    return frame

# Presence System   
def markAttendanceIntoDB(name,id):
    status="masuk"
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    dateString=now.strftime('%d-%m-%y')
    sq.insertRow(id,dateString,name,dtString,status)

# Doing some Face Recognition with the webcam

def gen_frame():
    listname = []
    listnim = []
    listid = []
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 1280)
    video_capture.set(4, 720)
    while True:
        _, frame = video_capture.read()
        gray = cv2.cvtColor(np.array(frame, dtype = 'uint8'), cv2.COLOR_BGR2GRAY)
        frameResized = cv2.resize(frame, size)
        grayResized = cv2.cvtColor(frameResized, cv2.COLOR_BGR2GRAY)
        frameCrop=getFrameCrop(gray,frame)
        listFrame = []
        listFrame.append(frameCrop)
        try:
                if _ and len(frameCrop) != 0:
                    frameCrop=cv2.resize(np.float32(frameCrop),size, interpolation=cv2.INTER_AREA)
                    print(len(frameCrop))
                    x = image.img_to_array(frameCrop)
                    x = np.expand_dims(x, axis=0)
                    images = preprocess_input(x)
                    classes = new_model.predict(images)
                    na=np.argmax(classes)
                    namaObjek=namaLabel[na]
                    namaObjek = str(namaObjek)
                    canvas= detect(gray, frame,namaObjek)
                    #cv2.imshow('Video', canvas)
                    print(namaObjek)
                    listname.append(namaObjek)
                    listid.append(str(na))
                    print(len(listname))
                    print(listname)
                    if len(listname) > 20:
                        markAttendanceIntoDB(mode(listname),mode(listid))
                        markAttendanceIntoCloud(namaObjek)
                        print("Kehadiran berhasil")
                        listname.clear()    
                        listFrame.clear()
        except Exception as e:
            print(str(e))
        if not _ :
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()