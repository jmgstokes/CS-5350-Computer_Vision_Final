#Name: Jeff Stokes
#Date: 12/06/2020
#Assignment: Final
#Instructor: Gil Gallegos, gil.gallegos@gmail.com
#TA: Rahim Ullah, rullah@live.nmhu.edu

import cv2
import numpy as np

#meanshift tracking method

#initialize camera
cap = cv2.VideoCapture('video_input/input0.avi')

#take first frame of video
ret, frame = cap.read()


#setup default window location
r, h, c, w = 20, 150, 60, 180
track_window = (c, r, w, h)

#crop region of interest for tracking
roi = frame[r:r+h, c:c+w]

#convert cropped window to HSV color space
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

#create mask between HSV bounds
lower_purple = np.array([125,0,0])
upper_purple = np.array([175,255,255])
mask = cv2.inRange(hsv_roi, lower_purple, upper_purple)

#obtain color histogram of ROI
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])

#normalize values to lie between 0 and 255
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

#set up termination criteria
#stop calculating centroid shift after 10 iterations or if centroid has moved 1+ pixels
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

i=0
while i<50:
	#read webcame frame
	ret, frame = cap.read()

	if ret == True:
		#convert to hsv
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		#calculate histogram back projection
		dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

		#apply meanshift to get new location
		ret, track_window = cv2.meanShift(dst, track_window, term_crit)

		#draw it on image
		x, y, w, h = track_window
		img2 = cv2.rectangle(frame, (x,y), (x+w, y+h), 255, 2)

		X, Y = int((x*2+w)/2), int((y*2+h)/2)
		cv2.circle(img2, (X,Y), 5, (250,50,50), -1)

		cv2.imshow('Meanshift Tracking', img2)
		cv2.imwrite('meanshift_centroids/frame_'+str(i)+'.jpg',img2)
		i+=1

		if cv2.waitKey(60) == 27: #Escape key
			break

	else:
		break
