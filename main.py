import cv2 as cv
import numpy as np
# import imutils

def aruco_detector():
    dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

    markerImage = np.zeros((200, 200), dtype=np.uint8)
    markerImage = cv.aruco.drawMarker(dictionary, 33, 200, markerImage, 1)

    cv.imwrite("marker33.png", markerImage)
    src = cv.imread("marker33_padded.png")

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    arucoParam = cv.aruco.DetectorParameters_create()
    bboxs, ids, rejected = cv.aruco.detectMarkers(gray, dictionary, parameters = arucoParam)
    pass


def blob_detector():
    import cv2
    import numpy as np;

    # Read image
    im = cv2.imread("yellow_blobs.png", cv2.IMREAD_GRAYSCALE)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = False
    params.minArea = 100

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)
    
def edge_detector_sobel():
    # Read the original image
    img = cv.imread('yellow_blobs.png')
    # Display original image
    cv.imshow('Original', img)
    cv.waitKey(0)

    # Convert to graycsale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)

    # # Sobel Edge Detection
    # sobelx = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    # sobely = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    # sobelxy = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    #
    # # Display Sobel Edge Detection Images
    # cv.imshow('Sobel X', sobelx)
    # cv.waitKey(0)
    # cv.imshow('Sobel Y', sobely)
    # cv.waitKey(0)
    # cv.imshow('Sobel X Y using Sobel() function', sobelxy)
    # cv.waitKey(0)

    # Canny Edge Detection
    edges = cv.Canny(image=img_blur, threshold1=75, threshold2=200)  # Canny Edge Detection
    # Display Canny Edge Detection Image
    cv.imshow('Canny Edge Detection', edges)
    cv.waitKey(0)

    cv.destroyAllWindows()

if __name__ == "__main__":
    edge_detector_sobel()
    pass