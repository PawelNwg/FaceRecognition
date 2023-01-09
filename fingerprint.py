import cv2
import numpy as np
import os

test_original = cv2.imread("100__M_Left_index_finger.tif")

for file in [file for file in os.listdir("Fingerprint_samples")]:
    fingerprint_database_image = cv2.imread("./Fingerprint_samples/" + file)
    sift = cv2.xfeatures2d.SIFT_create()

    keyPoints_1, descriptors_1 = sift.detectAndCompute(test_original, None)
    keyPoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)

    matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict()).knnMatch(descriptors_1, descriptors_2, k=2)
    match_points = []

    for p, q in matches:
        if p.distance < 0.1 + q.distance:
            match_points.append(p)
        keyPoints = 0
        if(len(keyPoints_1) <= len(keyPoints_2)):
            keyPoints = len(keyPoints_1)
        else:
            keyPoints = len(keyPoints_2)

    if (len(match_points) / keyPoints > 0.95):
        print("% match: ", keyPoints / len(match_points) * 100)
        print("Fingerprint Id " + str(file))
        result = cv2.drawMatches(test_original, keyPoints_1, fingerprint_database_image, keyPoints_2, match_points, None)
        result = cv2.resize(result, None, fx=4, fy=4)
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break