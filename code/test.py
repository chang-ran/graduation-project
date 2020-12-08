from skimage import io,color
import face_recognition
import numpy as np
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import cv2

src = '../data/tongue/tongue_dev_99648.jpg'
image = face_recognition.load_image_file(src)

def buildmask(shape, face_landmarks, facial_feature):
    h,w,_ = shape
    mask = np.zeros([h,w])
    cv2.fillPoly(mask,np.int32([face_landmarks[facial_feature]]),1)
    return mask

face_landmarks_list = face_recognition.face_landmarks(image)

print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

# Create a PIL imagedraw object so we can draw on the picture
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

for face_landmarks in face_landmarks_list:

    # Print the location of each facial feature in this image
    for facial_feature in face_landmarks.keys():
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    # Let's trace out each facial feature in the image with a line!
    for facial_feature in face_landmarks.keys():
        mask = buildmask(image.shape,face_landmarks,facial_feature)
        plt.imshow(mask)
        plt.show()

        # print(face_landmarks[facial_feature])
        # d.point(face_landmarks[facial_feature])      

# Show the picture
# pil_image.show()
