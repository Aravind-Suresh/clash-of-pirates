#
# /**
#  * @author {aravind}
#  * Created on 2016-02-27 11:18
#  *
#  * Clash of pirates source code
#  *
#  */
#

# Necessary modules imported
import numpy as np
import sys
import cv2
import math

# Usage for IMAGE MODE ( MODE = 1 ): python main.py <mode> <image input>
# Usage for VIDEO MODE ( MODE = 0 ): python main.py <mode>

# CAMERA_PORT_INDEX : Camera port index from which video is read
CAMERA_PORT_INDEX = 0
# MODE = 1 : image input, 0 : video input with VideoCapture(CAMERA_PORT_INDEX)
MODE = eval(sys.argv[1])
# DEBUG = 1 : displays images
DEBUG = 1
# LOG_RESULT = 1 : logs final deductions
LOG_RESULT = 1

def orderClockwise(ptsO, pt):
	pts = ptsO - np.asarray(pt)
	pts = np.array(pts, dtype=np.float32)
	slopes = []
	for p in pts:
		if p[0] > 0:
			slopes.append(math.atan(p[1]/p[0]))
		else:
			slopes.append(math.pi + math.atan(p[1]/p[0]))
	ptsSorted = [y for x, y in sorted(zip(list(slopes), list(np.arange(len(ptsO)))))]
	ptsSorted = ptsO[ptsSorted]
	return ptsSorted

def getColorFromHue(hue):
	if hue > 160 or hue < 50:
		return "RED"
	elif hue > 50 and hue < 100:
		return "GREEN"
	elif hue >= 100 and hue < 150:
		return "PURPLE"

def dist(pt1, pt2):
	return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

def detectShapes(img, imgHue):
	img = cv2.GaussianBlur(img, (5, 5), 0)
	ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	im, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	contours = sorted(contours, key=cv2.contourArea)
	#print contours
	#for cnt in contours[0:-2]:
	#	approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)

	# Largest is the sheet; 2nd largest is the square
	ctr = contours[-2]
	approx = cv2.approxPolyDP(ctr, 0.01*cv2.arcLength(ctr, True), True)

	bdry = np.array([[approx[2][0][0], approx[2][0][1]],
					[approx[1][0][0], approx[1][0][1]],
					[approx[0][0][0], approx[0][0][1]],
					[approx[3][0][0], approx[3][0][1]]], dtype="int0")
	# Square dimensions
	dimx = 400
	dimy = 400
	pts1 = np.float32(bdry)
	pts2 = np.float32([[0,dimx-1],[0,0],[dimy-1,0],[dimy-1,dimx-1]])

	M = cv2.getPerspectiveTransform(pts1, pts2)
	imgWarp = cv2.warpPerspective(img, M, (int(dimy),int(dimx)))
	MInv = cv2.getPerspectiveTransform(pts2, pts1)

	if DEBUG:
		cv2.imshow("imgWarp", imgWarp)

	# cv2.drawContours(img, [ctr], 0, (0,0,255), 2)

	cimg = imgWarp.copy()
	circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,20,
                            param1=40,param2=40,minRadius=0,maxRadius=0)
	circles = np.uint16(np.around(circles))

	ret, thresh_new = cv2.threshold(cimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	circlesArr = []
	X, Y = np.meshgrid(np.linspace(0, img.shape[1]-1, img.shape[1]), np.linspace(0, img.shape[0]-1, img.shape[0]))
	for i in circles[0,:]:
		# Covering circles
		cv2.circle(thresh_new, (i[0],i[1]), i[2]+10, 255, -1)
		cv2.circle(imgWarp, (i[0],i[1]), i[2], 127, -1)
		colorHue = np.mean(imgHue[np.where((X-i[0])**2 + (Y-i[1])**2 < i[2]**2)])
		# print colorHue
		coord = np.dot(MInv, np.asarray([i[0], i[1], 1]))
		coord = (np.float32(coord) / coord[2])[:2]
		circlesArr.append({"center": tuple(coord), "area":np.pi*i[2]*i[2], "color":getColorFromHue(colorHue)})

	im2, contours2, hierarchy2 = cv2.findContours(thresh_new,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	contours2 = sorted(contours2, key=cv2.contourArea)
	areaMax = 0.8*cv2.contourArea(contours2[-1])
	areaLargeSq = cv2.contourArea(contours2[-2])

	# 0.25 - preventing CAR to be detected as a square
	contours2_copy = contours2
	contours2 = filter(lambda x: cv2.contourArea(x) < areaMax and cv2.contourArea(x) > 0.25*areaLargeSq, contours2)
	contours3 = filter(lambda x: cv2.contourArea(x) > 0.05*areaLargeSq and cv2.contourArea(x) < 0.25*areaLargeSq, contours2_copy)

	#print contours3

	carPos = None
	if len(contours3):
		rect = cv2.minAreaRect(contours3[0])
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		coord = np.mean(box, 0)
		box = orderClockwise(box, coord)
		print box

		d1 = dist(box[0], box[1])
		d2 = dist(box[1], box[2])

		angle = None
		if d1 < d2:
			num = box[1][1] - box[2][1]
			den = box[1][0] - box[2][0]
			if den == 0:
				angle = 90
			angle = np.degrees(np.arctan(1.0*num/den))
		else:
			num = box[1][1] - box[0][1]
			den = box[1][0] - box[0][0]
			if den == 0:
				angle = 90
			angle = np.degrees(np.arctan(1.0*num/den))

		cv2.rectangle(imgWarp, tuple(box[0]), tuple(box[2]), 255,2)
		coord = np.dot(MInv, np.asarray([coord[0], coord[1], 1]))
		coord = (np.float32(coord) / coord[2])[:2]
		carPos = {"center": tuple(coord), "orientation": angle}

	squares = []
	for cnt in contours2:
		approx = cv2.approxPolyDP(cnt, 0.08*cv2.arcLength(cnt, True), True)
		if len(approx) == 4:
			bdry = np.array([[approx[2][0][0], approx[2][0][1]],
							[approx[1][0][0], approx[1][0][1]],
							[approx[0][0][0], approx[0][0][1]],
							[approx[3][0][0], approx[3][0][1]]], dtype="int0")
			# print np.mean(bdry, 0)
			coord = np.mean(bdry, 0)
			coord = np.dot(MInv, np.asarray([coord[0], coord[1], 1]))
			coord = (np.float32(coord) / coord[2])[:2]
			squares.append({"center": tuple(coord), "area": cv2.contourArea(approx)})
			cv2.drawContours(imgWarp, [cnt], 0, 255, -1)

	if DEBUG:
		cv2.imshow("img-final", imgWarp	)

	# FINAL INFO
	return carPos, squares, circlesArr

if MODE:
	imgClr = cv2.imread(sys.argv[2])
	imgHue = cv2.split(cv2.cvtColor(imgClr, cv2.COLOR_BGR2HSV))[0]
	img = cv2.cvtColor(imgClr, cv2.COLOR_BGR2GRAY)
	carPos, squares, circlesArr = detectShapes(img, imgHue)

	if DEBUG:
		cv2.imshow("img-hue", imgHue)
		cv2.imshow("img-clr", imgClr)
	if LOG_RESULT:
		print squares
		print circlesArr
		print carPos
	cv2.waitKey(0)
else:
	cap = cv2.VideoCapture(CAMERA_PORT_INDEX)
	while(1):
		ret, imgClr = cap.read()
		imgHue = cv2.split(cv2.cvtColor(imgClr, cv2.COLOR_BGR2HSV))[0]
		img = cv2.cvtColor(imgClr, cv2.COLOR_BGR2GRAY)
		carPos, squares, circlesArr = detectShapes(img, imgHue)
		k = cv2.waitKey(1)
		if k == 27:
			break
		if LOG_RESULT:
			print squares
			print circlesArr
			print carPos

cv2.destroyAllWindows()
