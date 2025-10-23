import cv2 as cv
imgTest = cv.imread("Lab1/test.jpg")

cv.imshow("Display windows", imgTest)
cv.waitKey(0)