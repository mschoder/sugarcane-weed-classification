import numpy as np
import cv2
import shutil
import os, copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

cap = cv2.VideoCapture('/home/mschoder/data/raw_video/Nightshade-2-LEFT_trim_67pct_speed.mp4')

# fourcc = cv2.VideoWriter_fourcc(*'h263')
out_size = (1200, 800)
# out_size = (704,576)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
<<<<<<< Updated upstream
out = cv2.VideoWriter('home/mschoder/data/raw_video/test_vid_1200x800.avi',fourcc, 20.0, out_size)
=======
out = cv2.VideoWriter('/Users/mschoder/weeding_project/processed_video/test_vid_1200x800.avi',fourcc, 20.0, out_size)
>>>>>>> Stashed changes

i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        # Rotate frame so movement is horizontal
        frame = np.rot90(frame)

        # Crop from center
        hp, wp, _ = frame.shape
        h, w = (wp*2/3, wp)
        center = (frame.shape[0]/2-400, frame.shape[1]/2)
        x = center[1] - w/2
        y = center[0] - h/2
        frame = frame[int(y):int(y+h), int(x):int(x+w)]

        # resize to desired output size
        frame = cv2.resize(frame, out_size)

        # write the output frame
        out.write(frame)
        i += 1

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # if i > 100:
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()