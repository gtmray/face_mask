import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('filename2.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         20, size)

while (cap.isOpened()):
    ret, img = cap.read()
    if ret == True:

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 100, 100), 10)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (50, 50)

        # fontScale
        fontScale = 1

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # img_size = 100
        #
        # img = cv2.resize(img, (img_size, img_size))
        # img = np.reshape(img, [1, img_size, img_size, 3])

        #classes = model.predict_classes(img)
        classes = 1
        if classes == 1:
            message = 'कृपया मास्क लाउनु होला। '
        else:
            message = 'धन्यबाद मास्क लाउनुभएकोमा। '

        # Using cv2.putText() method
        cv2.putText(img, message, org, font,
                    fontScale, color, thickness, cv2.LINE_AA)
        result.write(img)
    else:
        break
cap.release()
cv2.destroyAllWindows()
