import face_recognition
import glob
import os
import cv2
import numpy as np
known_face_encodings = []
known_face_names = []
frame_resizing = 0.25


#Return All Images in image Folder
images_path = glob.glob(os.path.join("images/", "*.*"))
for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            print(filename)
            if filename=='modi':
                filename="Modi Ji"
            elif filename=='shubham':
                filename='Shubham Kumar'
            known_face_names.append(filename)
            known_face_encodings.append(face_recognition.face_encodings(rgb_img)[0])

cam=cv2.VideoCapture(0)
while True:
    ret,frame=cam.read()
    small_frame = cv2.resize(frame, (0, 0), fx=frame_resizing, fy=frame_resizing)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    #cv2.imshow("Frames:",rgb_small_frame)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        #How Much Image Found
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
        face_locations = np.array(face_locations).astype(int)
        face_locations = face_locations / frame_resizing
    for face_loc,name in zip(face_locations,face_names):
        y1,x1,y2,x2=face_loc[0],face_loc[1],face_loc[2],face_loc[3]  
        cv2.putText(frame,name,(int(x2)-10, int(y1)-30),cv2.FONT_HERSHEY_DUPLEX,1,(255, 255, 255),2, cv2.LINE_AA)
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,0),2)
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1) 
    if key==27:
        break
cam.release()
cv2.destroyAllWindows()



