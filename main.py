import cv2 as cv
import numpy as np
import imutils

dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

markerImage = np.zeros((200, 200), dtype=np.uint8)
markerImage = cv.aruco.drawMarker(dictionary, 33, 200, markerImage, 1)

cv.imwrite("marker33.png", markerImage)
src = cv.imread("marker33_padded.png")

gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
arucoParam = cv.aruco.DetectorParameters_create()
bboxs, ids, rejected = cv.aruco.detectMarkers(gray, dictionary, parameters = arucoParam)
pass
