import numpy as np
import cv2
#from google.colab.patches import cv2_imshow

model = models.load_model('model.h5')
cap = cv2.VideoCapture('/content/vdo.mp4')
vid_name = 'try3.avi'

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# With webcam get(CV_CAP_PROP_FPS) does not work.
# Let's see for ourselves.

if int(major_ver)  < 3 :
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 
   
# Below VideoWriter object will create 
# a frame of above defined The output  
# is stored in 'filename.avi' file. 
result = cv2.VideoWriter(vid_name,  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         30, size) 
msg = []
while(cap.isOpened()):
    ret, img = cap.read()
    if ret==True: 
      font = cv2.FONT_HERSHEY_SIMPLEX 
    
      # org 
      org = (50, 50) 
        
      # fontScale 
      fontScale = 1
        
      # Blue color in BGR 
      color = (200, 100, 0) 
        
      # Line thickness of 2 px 
      thickness = 1
      
      img_size = 100

      img_prd = cv2.resize(img,(img_size,img_size))    
      img_prd = np.reshape(img_prd,[1,img_size,img_size,3])

      classes = model.predict_classes(img_prd)
      if classes==1:
        #message = 'कृपया मास्क लाउनु होला। '
        message = 'Mask laaune gar khaate'
      elif classes==0:
        #message = 'धन्यबाद मास्क लाउनुभएकोमा। '
        message = 'sahi ho mask laaunu parxa'
      else:
        message = 'waiting!!!!!!!!!!!!'
      # Using cv2.putText() method
      msg.append(message) 
      cv2.putText(img, message, org, font,  
                        fontScale, color, thickness, cv2.LINE_AA)
      #result.write(img)  
    else:
      break
cap.release()
cv2.destroyAllWindows()
