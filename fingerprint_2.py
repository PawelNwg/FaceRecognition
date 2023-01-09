import cv2
import numpy as np
import os

test_original = cv2.imread("pobrane.bmp")

sift = cv2.xfeatures2d.SIFT_create()
keyPoints_1, descriptors_1 = sift.detectAndCompute(test_original, None)
# keypoint_1 to wektor
# Here kp will be a list of keypoints and des is a numpy array of shape (Number of Keypoints)×128.

result = cv2.drawKeypoints(test_original, keyPoints_1, test_original)
result = cv2.resize(result, None, fx=4, fy=4)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()