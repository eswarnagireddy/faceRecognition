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

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)    
    


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
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
      
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
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


        # Display the results
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

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()


import pandas as pd

t=[0,0,0,0,0,]
R=['101','102','103','104','105']
n=[ 'Munna','Anu','Obama','Trump','Tony Stark']
a=[0,0,0,0,0]
for i in range(len(present)):
    if present[i] in n: 
        a.pop(n.index(present[i]))
        a.insert(n.index(present[i]),'P')
        t.pop(n.index(present[i]))
        t.insert(n.index(present[i]),c[i])
for _ in range(len(a)):
        if a[_]==0:
            a.pop(_)
            a.insert(_,"-")
            t.pop(_)
            t.insert(_,"-")
data = {'Roll No.':R,'Name':n,'P/A':a,'Time':t}

df = pd.DataFrame(data)
df=df[['Roll No.','Name','P/A','Time']]

from IPython.display import HTML

def hover(hover_color="#ffff99"):
    return dict(selector="tr:hover",
                props=[("background-color", "%s" % hover_color)])

styles = [
    hover(),
    dict(selector="th", props=[("font-size", "150%"),
                               ("text-align", "center")]),
    dict(selector="caption", props=[("caption-side", "center")])
]
def highlight(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    z=[]
    for i in range(len(df.index)):
         if df.iloc[i,2]=='P':
                z.append(i)
         else:
                z.append(-1)     
    
    return ['background-color: greenyellow' if i==z[i] else ''for i in range(len(z))]
html = (df.style.set_table_styles(styles)
          .set_caption("Attendence Sheet.").set_properties(**{'font-size':'11pt'})
           .apply(highlight))

html