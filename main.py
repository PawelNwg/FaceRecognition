import cv2

face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth = cv2.CascadeClassifier("Mouth.xml")

# scale - określa +- jaki rozmiar obiektu ma byc wykrywany
# minNeighbors - ile musi zostać wykrytych obiektów na obszarze żeby uznać to za np twarz im wiecej tym wieksza dokładnosc
# teoria działania:
# https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    cords = []
    for(x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        cv2.putText(img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        cords = [x, y, w, h]

    return cords

def detect(img, color):

    cords = draw_boundary(img, face, 1.1, 10, color, "Face")
    if(len(cords) == 4):
        cropped_img = img[cords[1]:cords[1] + cords[3], cords[0]:cords[0] + cords[2]]
        cords = draw_boundary(cropped_img, eyes, 1.1, 14, color, "Eye")
        cords = draw_boundary(cropped_img, mouth, 1.3, 30, color, "Mouth")

    return img

# Capturing real time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
video_capture = cv2.VideoCapture(0)

while True:
    # Reading image from video stream
    _, img = video_capture.read()
    # Writing processed image in a new window
    img = detect(img, (255, 0, 0))

    cv2.imshow("Biometria II", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing web-cam
video_capture.release()
# Destroying output window
cv2.destroyAllWindows()