import cv2
from tensorflow.keras import models
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = models.load_model('model.h5')


## Make canvas and set the color
img = np.zeros((200,400,3),np.uint8)
b,g,r,a = 0,255,0,0

fontpath = "ArialUnicodeMS.ttf"
font_nep = ImageFont.truetype(fontpath, 32)


# To capture video from webcam.
cap = cv2.VideoCapture(0)
org = (50, 50)

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

    # color in BGR
    color = (36, 255, 12)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        crop_img = img[y:y+h, x:x+w]

    if len(faces)!= 0:
        img_for_pred = crop_img
        org = (x, y - 50)

    else:
        img_for_pred = img

    font = cv2.FONT_HERSHEY_SIMPLEX



    # fontScale
    fontScale = 0.9

    # Line thickness of 2 px
    thickness = 2

    img_size = 100

    img_pred = cv2.resize(img_for_pred, (img_size, img_size))
    img_pred = np.reshape(img_pred, [1, img_size, img_size, 3])

    classes = model.predict_classes(img_pred)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    if classes == 1:
        #message = 'Masks laa khaate'
        message = 'कृपया मास्क लाउनु होला। '
    else:
        #message = 'Sahi ho masks laaunu parxa'
        message = 'धन्यबाद मास्क लाउनुभएकोमा। '

    # cv2.putText(img, message, org, font,
    #             fontScale, color, thickness, cv2.LINE_AA)
    draw.text(org, message, font=font_nep, fill=(b, g, r, a))
    img = np.array(img_pil)

    # Display
    cv2.imshow('img', img)

    cv2.putText(img, message, org, font,
                fontScale, color, thickness, cv2.LINE_AA)


    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break




# Release the VideoCapture object
cap.release()
