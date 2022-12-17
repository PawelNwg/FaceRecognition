from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import imutils
import time
import cv2

input = 'faces/'
video_capture = cv2.VideoCapture(0)

# Wytrenowany model twarzy
face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def get_coordinates(image, scaleFactor, minNeighbors):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = face.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    cords = []
    for (x, y, w, h) in features:
        cords = (x, y, x + w, y + h)

    return cords


def detect_faces(image):
    coordinates = get_coordinates(image, 1.1, 10)
    return [coordinates]


def load_face_dataset(inputPath, minSamples=3):
    imagePaths = list(imutils.paths.list_images(inputPath))

    # Get people names from images
    splitNames = [p.split('/')[-1].split('_')[0] for p in imagePaths]
    (uniqueNames, counts) = np.unique(splitNames, return_counts=True)
    namesList = uniqueNames.tolist()

    faces = []
    labels = []

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        name = imagePath.split('/')[-1].split('_')[0]

        if counts[namesList.index(name)] < minSamples:
            continue

        facesCoordinates = detect_faces(image)

        for (startX, startY, endX, endY) in facesCoordinates:
            faceROI = image[startY:endY, startX:endX]
            faceROI = cv2.resize(faceROI, (47, 62))
            faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)

            faces.append(faceROI)
            labels.append(name)

    faces = np.array(faces)
    labels = np.array(labels)

    return faces, labels


def get_face(img):
    boxes = detect_faces(img)
    if (len(boxes) == 1 and not boxes[0]) or not boxes:
        return (img, False)

    for (startX, startY, endX, endY) in boxes:
        faceROI = img[startY:endY, startX:endX]
        faceROI = cv2.resize(faceROI, (47, 62))
        faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)

    return (faceROI, True)


# Load the CALTECH faces dataset
print("[INFO] Loading dataset...")
(faces, labels) = load_face_dataset(input, minSamples=2)
print("[INFO] {} images in dataset".format(len(faces)))

# Flatten all 2D faces into a 1D list of pixel intensities
pcaFaces = np.array([f.flatten() for f in faces])

le = LabelEncoder()
labels = le.fit_transform(labels)

# Construct our training split
(_, _, trainX, _, _, _) = train_test_split(faces, pcaFaces, labels, test_size=0.5, stratify=labels, random_state=42)

# Compute the PCA (eigenfaces) representation of the data, then
# project the training data onto the eigenfaces subspace
print("[INFO] Creating eigenfaces...")
pca = PCA(svd_solver="randomized", n_components=2, whiten=True)

start = time.time()
pca.fit_transform(trainX)
end = time.time()
print("[INFO] Computing eigenfaces took {:.4f} seconds".format(end - start))

print("[INFO] Training classifier...")
model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42)
model.fit(pca.fit_transform(pcaFaces), labels)

while True:
    ret, img = video_capture.read(0)
    if not ret:
        continue
    (face_img, success) = get_face(img)
    if not success:
        continue

    face_array = np.asarray([face_img.flatten()])
    prediction = model.predict(pca.transform(face_array))

    # Grab the predicted name and actual name
    predictedName = le.inverse_transform([prediction[0]])[0]
    cv2.putText(img, "Osoba: {}".format(predictedName), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Biometria - Zadanie 3", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
