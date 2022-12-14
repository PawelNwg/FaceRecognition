from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import imutils
import time
import cv2
import math

name = "Samples/%d.jpg"
sample_count = 0
video_capture = cv2.VideoCapture(0)

# wytrenowane modele
face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth = cv2.CascadeClassifier("Mouth.xml")


# metoda rysuje kształt prostokąta, wykrywając obiekt detektorem
# classifier - zadany klasyfikator
# scale factor - skalowanie obrazu wejściowego
# minNeighbors - ile puntków należy zidentyfikować w obszarze aby zaliczyć obiekt jako wykryty
def draw_boundary(img, classifier, scaleFactor, minNeighbors):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    cords = []
    for(x, y, w, h) in features:
        cords = (x, y, x + w, y + h)

    return cords

# metoda dokonuje detekcji twarzy, wycina ten obszar z obrazu a następnie
# na wycinku uruchamia detektory oczu oraz ust
def detect_faces(img):
    global sample_count
    cords = draw_boundary(img, face, 1.1, 10)
    if(len(cords) == 4):
        cropped_img = img[cords[1] - 30 :cords[1] + cords[3] + 30, cords[0] - 30:cords[0] + cords[2] + 30]
        if sample_count % 30 == 0:
            cv2.imwrite(name % math.ceil(sample_count/30.0), cropped_img)  # save frame as JPEG file
        sample_count += 1
    return [cords]


def load_face_dataset(inputPath, minSamples=3):
    # grab the paths to all images in our input directory, extract
    # the name of the person (i.e., class label) from the directory
    # structure, and count the number of example images we have per
    # face
    imagePaths = list(imutils.paths.list_images(inputPath))
    names = [p.split('/')[-1].split('_')[0] for p in imagePaths]
    (names, counts) = np.unique(names, return_counts=True)
    names = names.tolist()
    # initialize lists to store our extracted faces and associated
    # labels
    faces = []
    labels = []
    # loop over the image paths
    for imagePath in imagePaths:
        # load the image from disk and extract the name of the person
        # from the subdirectory structure
        image = cv2.imread(imagePath)
        name = imagePath.split('/')[-1].split('_')[0]
        # only process images that have a sufficient number of
        # examples belonging to the class
        if counts[names.index(name)] < minSamples:
            continue

        # perform face detection
        boxes = detect_faces(image)
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # extract the face ROI, resize it, and convert it to
            # grayscale
            faceROI = image[startY:endY, startX:endX]
            faceROI = cv2.resize(faceROI, (47, 62))
            faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
            # update our faces and labels lists
            faces.append(faceROI)
            labels.append(name)
    # convert our faces and labels lists to NumPy arrays
    faces = np.array(faces)
    labels = np.array(labels)
    # return a 2-tuple of the faces and labels
    return (faces, labels)

def get_face(img):
    boxes = detect_faces(img)
    if (len(boxes) == 1 and not boxes[0]) or not boxes:
        return (img, False)

    for (startX, startY, endX, endY) in boxes:
        faceROI = img[startY:endY, startX:endX]
        faceROI = cv2.resize(faceROI, (47, 62))
        faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)

    return (faceROI, True)

input = 'faces/'
confidence = 0.5
num_components = 2
visualize = (1 == 1)

# load the CALTECH faces dataset
print("[INFO] loading dataset...")
(faces, labels) = load_face_dataset(input, minSamples=2)
print("[INFO] {} images in dataset".format(len(faces)))

# flatten all 2D faces into a 1D list of pixel intensities
pcaFaces = np.array([f.flatten() for f in faces])

# encode the string labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# construct our training and testing split
split = train_test_split(faces, pcaFaces, labels, test_size=0.33, stratify=labels, random_state=42)
(origTrain, origTest, trainX, testX, trainY, testY) = split

# compute the PCA (eigenfaces) representation of the data, then
# project the training data onto the eigenfaces subspace
print("[INFO] creating eigenfaces...")
pca = PCA(
    svd_solver="randomized",
    n_components=num_components,
    whiten=True)
start = time.time()
trainX = pca.fit_transform(trainX)
end = time.time()
print("[INFO] computing eigenfaces took {:.4f} seconds".format(
    end - start))

print("[INFO] training classifier...")
model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42)
model.fit(trainX, trainY)
# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(pca.transform(testX))
print(classification_report(testY, predictions, target_names=le.classes_))

# check to see if the PCA components should be visualized
while True:
    ret, img = video_capture.read(0)
    if not ret:
        continue
    (face_img, success) = get_face(img)
    if not success:
        continue
    face_array = np.asarray([face_img.flatten()])
    prediction = model.predict(pca.transform(face_array))
    # grab the predicted name and actual name
    predName = le.inverse_transform([prediction[0]])[0]
    # draw the predicted name and actual name on the image
    cv2.putText(img, "pred: {}".format(predName), (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # display the predicted name  and actual name
    print("[INFO] prediction: {}".format(predName))

    cv2.imshow("Biometria-ZAD3", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing web-cam
video_capture.release()
# Destroying output window
cv2.destroyAllWindows()
