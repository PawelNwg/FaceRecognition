import uuid

import cv2
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import mediapipe as mp
import time as time
import math
sample_count = 0
#wytrenowane modele
#modele OpenCV
face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth = cv2.CascadeClassifier("Mouth.xml")
#modele mediapipe
#mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
#mp_face_mesh = mp.solutions.face_mesh
#face_mesh = mp_face_mesh.FaceMesh()

#segmentor = SelfiSegmentation()
# pobieranie obrazu z kamery, 0 oznacza kamere wbudowana
video_capture = cv2.VideoCapture(0)

# zakres kolorow skory w hsv
#lower = np.array([0, 30, 53], dtype = "uint8")
#upper = np.array([20, 180, 255], dtype = "uint8")

#lower2 = np.array([172, 30, 53], dtype = "uint8")
#upper2 = np.array([180, 180, 210], dtype = "uint8")

#wybrene id landmarkow do pomiaru predkosci
#landmarks_ids = [1, 15, 50, 280]

# metoda rysuje kształt prostokąta, wykrywając obiekt detektorem
# classifier - zadany klasyfikator
# scale factor - skalowanie obrazu wejściowego
# minNeighbors - ile puntków należy zidentyfikować w obszarze aby zaliczyć obiekt jako wykryty
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    cords = []
    for(x, y, w, h) in features:
        #cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        #cv2.putText(img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        cords = [x, y, w, h]

    return cords

# metoda dokonuje detekcji twarzy, wycina ten obszar z obrazu a następnie
# na wycinku uruchamia detektory oczu oraz ust
def detect(img, color):
    cords = draw_boundary(img, face, 1.1, 10, color, "Face")
    global sample_count
    if(len(cords) == 4):
        cropped_img = img[cords[1] - 30 :cords[1] + cords[3] + 30, cords[0] - 30:cords[0] + cords[2] + 30]
        if sample_count % 30 == 0:
            cv2.imwrite("Samples/frame%d.jpg" % math.ceil(sample_count/30.0), cropped_img)  # save frame as JPEG file
        cords = draw_boundary(cropped_img, eyes, 1.1, 14, color, "Eye")
        cords = draw_boundary(cropped_img, mouth, 1.3, 30, color, "Mouth")
        sample_count += 1
    return img

# pozycje wybranych landmarkow
#landmarks_prev_position = {landmarks_ids[0] : None, landmarks_ids[1]: None, landmarks_ids[2]: None, landmarks_ids[3]: None }
#landmarks_prev_time = {landmarks_ids[0] : None, landmarks_ids[1]: None, landmarks_ids[2]: None, landmarks_ids[3]: None }

while True:
    ret, img = video_capture.read(0)
    image = img.copy()
    # Rysowanie konturów
    #if ret:
        #image = segmentor.removeBG(image, (0, 0, 0))
    imgInHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #mask1 = cv2.inRange(imgInHSV, lower, upper)
    #mask2 = cv2.inRange(imgInHSV, lower2, upper2)

    #Gaussian Blur - rozmycie gaussowskie - dodanie do maski
    #mask1 = cv2.GaussianBlur(mask1, (3, 3), 0)
    #mask2 = cv2.GaussianBlur(mask2, (3, 3), 0)

    #imgWithSkin1 = cv2.bitwise_and(image, image, mask = mask1)
    #imgWithSkin2 = cv2.bitwise_and(image, image, mask = mask2)

    #wzięcie obu zakresów kolorów skóry
    #skinImg = cv2.bitwise_or(imgWithSkin1, imgWithSkin2)

    #img_gray = cv2.cvtColor(skinImg, cv2.COLOR_BGR2GRAY)
    # wybranie tylko tych kolorów skóry (wartosc mniejsza niz 50 jest ustawiane na 0)
    #ret, thresh = cv2.threshold(img_gray, 25, 255, cv2.THRESH_BINARY)

    # rysowanie konturów na podstawie punktów
    #contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # rysowanie kontur
    # image_copy = skinImg.copy()
    #image_copy = img.copy()
    #cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
    #                 lineType=cv2.LINE_AA)

    # Wykrywanie oczu i ust
    img2 = img.copy()
    img2 = detect(img2, (255, 0, 0))

    # Mierzenie predkosci
    image3 = img.copy()
    image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
    #results = face_mesh.process(image3)

    cTime = time.time()
    # if results.multi_face_landmarks:
    #     for face_landmarks in results.multi_face_landmarks:
    #         mp_drawing.draw_landmarks(
    #             image=image3,
    #             landmark_list=face_landmarks,
    #             connections=mp_face_mesh.FACEMESH_FACE_OVAL,
    #             landmark_drawing_spec=None,
    #             connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
    #         )
    #         for id, landmark in enumerate(face_landmarks.landmark):
    #             ih, iw, ic = image3.shape
    #             x, y = int(landmark.x*iw), int(landmark.y*ih)
    #             if id in landmarks_ids:
    #                 cv2.putText(image3, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    #                 if landmarks_prev_time[id]:
    #                     time_difference = cTime - landmarks_prev_time[id]
    #                     smth = math.sqrt((x - landmarks_prev_position[id][0])**2 + (y - landmarks_prev_position[id][1])**2)/ time_difference
    #                     cv2.putText(image3, str(id) +  " " + str(smth), (20, 70*(landmarks_ids.index(id) + 1)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    #                 landmarks_prev_time[id] = cTime
    #                 landmarks_prev_position[id] = (x, y)

    cv2.imshow("Biometria II", np.hstack([np.vstack([img, img2]), np.vstack([img2, img2])]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing web-cam
video_capture.release()
# Destroying output window
cv2.destroyAllWindows()