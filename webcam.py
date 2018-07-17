# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:23:13 2018

@author: Phanidhar
"""

import face_recognition
import cv2
import pickle
from sklearn import neighbors
import time



video_capture = cv2.VideoCapture(0)


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
present=[]
c=[]
process_this_frame = True
if __name__ == "__main__":
    #knn_clf = train("knn_examples/train")
    knn_clf = pickle.load(open("knn_model.sav", 'rb'))
    
    while True:

        ret, frame = video_capture.read()

  
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
         
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            if (len(face_encodings)>0):
                
                closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)


                is_recognized = [closest_distances[0][i][0] <= 0.5 for i in range(len(face_locations))]

                face_names = []
                for pred, loc, rec in zip(knn_clf.predict(face_encodings),face_locations, is_recognized):
                     if rec:
                         face_names.append(pred)
                         if pred not in present: 
                                present.append(pred)
                                c.append(time.strftime('%H:%M:%S', time.localtime()))
                         
                         
                        
                     else :
                         face_names.append("unknown")
                 
        

        process_this_frame = not process_this_frame



        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            #print(name)
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
      # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'x' in Keyboard to exit the console
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
        
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

