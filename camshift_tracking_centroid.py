#Name: Jeff Stokes
#Date: 12/06/2020
#Assignment: Final
#Instructor: Gil Gallegos, gil.gallegos@gmail.com
#TA: Rahim Ullah, rullah@live.nmhu.edu

import cv2
import numpy as np

#initialize video
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
lower_bound = np.array([0,60,32])
upper_bound = np.array([180,255,255])
mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)

#obtain color histogram of ROI
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])

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
		#each pixel's value is it's probability
		dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

		#apply Camshift to get new location
		ret, track_window = cv2.CamShift(dst, track_window, term_crit)		

		#draw it on image, using polylines to represent Adaptive box
		pts = cv2.boxPoints(ret)
		pts = np.int0(pts)
		img2 = cv2.polylines(frame, [pts], True, 255, 2)

		#find center x,y coordinate of tracking window
		X = int(sum(pts[:,0])/4)
		Y = int(sum(pts[:,1])/4)

		#add centroid to tracking box
		cv2.circle(img2, (X,Y), 5, (250,50,50), -1)

		cv2.imshow('Camshift Tracking', img2)
		cv2.imwrite('camshift_centroids/frame_'+str(i)+'.jpg',img2)
		i+=1

		if cv2.waitKey(60) == 27: #Escape key
			break


	else:
		break
cv2.destroyAllWindows()
cap.release()