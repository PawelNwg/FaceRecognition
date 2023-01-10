import cv2
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from skimage.morphology import skeletonize

X = []
y = []

for file in [file for file in os.listdir("min1")]:
    fingerprint_database_image = cv2.imread("./min1/" + file)
    fingerprint_database_image = cv2.bitwise_not(fingerprint_database_image)
    noColor = cv2.cvtColor(fingerprint_database_image, cv2.COLOR_BGR2GRAY)
    _, noColor = cv2.threshold(noColor, 128, 1, cv2.THRESH_BINARY)
    X.append(np.ravel(noColor))
    y.append(1)

for file in [file for file in os.listdir("min2")]:
    fingerprint_database_image = cv2.imread("./min2/" + file)
    fingerprint_database_image = cv2.bitwise_not(fingerprint_database_image)
    noColor = cv2.cvtColor(fingerprint_database_image, cv2.COLOR_BGR2GRAY)
    _, noColor = cv2.threshold(noColor, 128, 1, cv2.THRESH_BINARY)
    X.append(np.ravel(noColor))
    y.append(2)

for file in [file for file in os.listdir("min3")]:
    fingerprint_database_image = cv2.imread("./min3/" + file)
    fingerprint_database_image = cv2.bitwise_not(fingerprint_database_image)
    noColor = cv2.cvtColor(fingerprint_database_image, cv2.COLOR_BGR2GRAY)
    _, noColor = cv2.threshold(noColor, 128, 1, cv2.THRESH_BINARY)
    X.append(np.ravel(noColor))
    y.append(3)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)

#szkieletyzacja
image = cv2.imread("fingerprints/cropped_fingerprint.tif")
skeleton = skeletonize(image)
skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
_, noColor = cv2.threshold(skeleton, 128, 1, cv2.THRESH_BINARY)

#przechodzenie po obrazkach
coordinates = []

rows = noColor.shape[0]
columns = noColor.shape[1]
for i in range(rows - 5):
    for j in range(columns - 5):
        subArray = noColor[i:(i+5), j:(j + 5)]
        prediction = clf.predict([np.ravel(subArray)])
        if prediction[0] == 1 or prediction[0] == 2:
            coordinates.append((i, j))

print(coordinates)
selected_coordinates = coordinates[0::200]
for coordinate in selected_coordinates:
    cv2.rectangle(skeleton, (coordinate[0], coordinate[1]), (coordinate[0] + 5, coordinate[1] + 5), (255, 0, 0), 1)

result = cv2.resize(skeleton, None, fx=5, fy=5)
cv2.imshow("Biometria IV - Odcisk palca", result)
cv2.waitKey(0)

