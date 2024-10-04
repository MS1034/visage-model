from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import os
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

detector = MTCNN()
model = load_model('Models/facenet_keras.h5')


face_print_registry_database = './Registry_database/Face Print/'
temp_face_capture_path = 'C:/Tensorflow_GPU/temp_cap/'


def marker_function(features, image, x1, x2, y1, y2):
    """ This function just mark the face and its components with a green line """

    detected_faces = features

    mask_clr = 'lawngreen'
    mask_width = 3
    draw = ImageDraw.Draw(image)
    draw.rectangle(((x1, y1), (x2, y2)), outline=mask_clr, width=mask_width)

    leye = detected_faces['keypoints']['left_eye']
    reye = detected_faces['keypoints']['right_eye']
    nose = detected_faces['keypoints']['nose']
    lmouth = detected_faces['keypoints']['mouth_left']
    rmouth = detected_faces['keypoints']['mouth_right']

    # left upper corner to left eye
    draw.line((x1, y1)+leye, fill=mask_clr, width=mask_width)
    # right bottom corner to right eye
    draw.line((x2, y2)+reye, fill=mask_clr, width=mask_width)
    # left bottom corner to left mouth
    draw.line((x1, y2)+lmouth, fill=mask_clr, width=mask_width)
    # right bottom corner to right  mouth
    draw.line((x2, y2)+rmouth, fill=mask_clr, width=mask_width)
    # right upper corner to right mouth
    draw.line((x2, y1)+reye, fill=mask_clr, width=mask_width)
    # left bottom corner to left eye
    draw.line((x1, y2)+leye, fill=mask_clr, width=mask_width)
    # left upper corner to left mouth
    draw.line((x1, y1)+lmouth, fill=mask_clr, width=mask_width)
    # right uupper corner to right mouth
    draw.line((x2, y1)+rmouth, fill=mask_clr, width=mask_width)
    # right upper corner to left eye
    draw.line((x2, y1)+leye, fill=mask_clr, width=mask_width)
    # left upeer corner to right eye
    draw.line((x1, y1)+reye, fill=mask_clr, width=mask_width)
    # left bottom corner to right mouth
    draw.line((x1, y2)+rmouth, fill=mask_clr, width=mask_width)
    # right bottom corner to left mouth
    draw.line((x2, y2)+lmouth, fill=mask_clr, width=mask_width)
    # right middle divisor to right eye and mouth
    draw.line((x2, (y1+y2)//2)+reye, fill=mask_clr, width=mask_width)
    draw.line((x2, (y1+y2)//2)+rmouth, fill=mask_clr, width=mask_width)
    # left middle divisor to left mouth and eye
    draw.line((x1, (y1+y2)//2)+leye, fill=mask_clr, width=mask_width)
    draw.line((x1, (y1+y2)//2)+lmouth, fill=mask_clr, width=mask_width)

    # top middle diviro to bot eyes
    draw.line(((x1+x2)//2, y1)+reye, fill=mask_clr, width=mask_width)
    draw.line(((x1+x2)//2, y1)+leye, fill=mask_clr, width=mask_width)
    # bottom middle divisor for bot mouths
    draw.line(((x1+x2)//2, y2)+rmouth, fill=mask_clr, width=mask_width)
    draw.line(((x1+x2)//2, y2)+lmouth, fill=mask_clr, width=mask_width)
    # middle star symbol
    draw.line(((x1+x2)//2, y1)+nose, fill=mask_clr, width=mask_width)
    draw.line(nose+(x2, y2), fill=mask_clr, width=mask_width)
    draw.line(nose+(x1, y2), fill=mask_clr, width=mask_width)

    draw.line(leye+reye, fill=mask_clr, width=mask_width)

    draw.line(leye+nose, fill=mask_clr, width=mask_width)

    draw.line(reye+nose, fill=mask_clr, width=mask_width)

    draw.line(nose+lmouth, fill=mask_clr, width=mask_width)

    draw.line(nose+rmouth, fill=mask_clr, width=mask_width)

    draw.line(lmouth+rmouth, fill=mask_clr, width=mask_width)

    eye_diffrence = reye[0]-leye[0]
#         print(eye_diffrence)

    image.save('marked.jpeg', "JPEG")


def get_face(path, resize_scale=(160, 160)):
    """ 
    This function is used to extract the face of 160x160 from the given image 

    Input : image directory path as input
    Returns : A tuple containing list of faces found, the original image.

    """

    face_list = list()

    image = Image.open(path)
    image = image.convert('RGB')

    pixels = np.asarray(image)
#     print(pixels.shape)

    faces_detected = detector.detect_faces(pixels)

    for detected_faces in faces_detected:
        x1, y1, width, height = detected_faces['box']
        components = detected_faces
        x1, y1 = abs(x1), abs(y1)

        x2, y2 = x1+width, y1+height

        final_face = pixels[y1:y2, x1:x2]

#         marker_function(detected_faces,image,x1,x2,y1,y2)
        marker_function(detected_faces, image, x1, x2, y1, y2)

        pic = Image.fromarray(final_face)
        pic = pic.resize(resize_scale)

        face_array = np.asarray(pic)
#         print(f'Extracted : {face_array.shape}')

        face_list.append(face_array)
    plt.imshow(image)
    plt.show()

    return face_list, image


def get_face_embeddings(face_pixels):

    face_pixels = face_pixels.astype('float32')

    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]
