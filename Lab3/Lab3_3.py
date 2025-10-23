#Lab3_3 Geometric Transformations of Images
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('Lab3/img2.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
 
res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
 
#OR
 
height, width = img.shape[:2]
res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)

cv.imshow("Resize",res)

#2. Сдвиг
img = cv.imread('Lab3/img2.jpg', cv.IMREAD_GRAYSCALE)
rows,cols = img.shape

M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img,M,(cols,rows))

cv.imshow('Translation',dst)

#3.Вращение
# cols-1 и rows-1 — это границы координат.
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
dst = cv.warpAffine(img,M,(cols,rows))

cv.imshow('Rotation',dst)

#Affine Transformation

img = cv.imread('Lab3/img2.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
rows,cols,ch = img.shape
 
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
 
M = cv.getAffineTransform(pts1,pts2)
 
dst = cv.warpAffine(img,M,(cols,rows))
 
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

#Perspective Transformation

img = cv.imread('Lab3/sudoku.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
rows,cols,ch = img.shape
 
pts1 = np.float32([[50,53],[267,53],[24,270],[289,270]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
 
M = cv.getPerspectiveTransform(pts1,pts2)
 
dst = cv.warpPerspective(img,M,(300,300))
 
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()