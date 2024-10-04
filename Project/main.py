from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace
import os
import cv2
from collections import defaultdict
from PIL import Image
import numpy as np
from camera_facemesh import capture_frame
from marker import marker_function
from utils import copy_images, images_in_folder
import shutil


backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'fastmtcnn',
    'retinaface',
    'mediapipe',
    'yolov8',
    'yunet',
    'centerface',
]

models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
    "GhostFaceNet",
]


def show_detected_face(face, image):

    cv2.imshow('Detected Face', face['face'])
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close the window after display

    x1, y1, width, height = face['facial_area']['x'], face['facial_area'][
        'y'], face['facial_area']['w'], face['facial_area']['h']
    left_eye = face['facial_area']['left_eye']
    right_eye = face['facial_area']['right_eye']

    x1, y1 = abs(x1), abs(y1)

    x2, y2 = x1+width, y1+height
    marker_function(Image.open(
        image), x1, x2, y1, y2, left_eye, right_eye)
    cv2.imshow('Detected Face', face['face'])
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close the window after display


def separate_faces_in_image(folder_path, area_threshold=1000):
    """
    Finds All faces inside images within a specified folder.

    Args:
        folder_path (str): The path to the folder containing the image files.

    Returns:
        list: A Face with the image that contain it and False if it is not detected
    """
    images = images_in_folder(folder_path)
    if not os.path.exists(unique_face_folder):
        os.makedirs(unique_face_folder)
    i = 0
    face_list = []
    for image in images:
        detected_faces = DeepFace.extract_faces(
            image, detector_backend=backends[0], enforce_detection=False)
        for face in detected_faces:

            # show_detected_face(face,image)

            x1, y1, width, height = face['facial_area']['x'], face['facial_area'][
                'y'], face['facial_area']['w'], face['facial_area']['h']
            face_area = width * height

            if (face_area < area_threshold):
                continue
            face_img = face['face']
            if (face['confidence'] > 0.95):
                continue
            face_image_uint8 = (face_img * 255).astype(np.uint8)

            cv2.imwrite(
                f"{unique_face_folder}/face_{i}.jpg", face_image_uint8)
            face_list.append(
                [f"{unique_face_folder}/face_{i}.jpg", image, False])
            i += 1

    return face_list


def consolidate_faces_to_images(face_list):
    """
    Consolidates images containing the same faces into a dictionary where keys are indices
    of the images and values are lists of paths to images containing the same face.

    Args:
        face_list (list): A list of tuples containing image paths, face embeddings, and detection flags.

    Returns:
        dict: A dictionary where keys are indices of the images and values are lists of paths
        to images containing the same face.
    """
    face_images = defaultdict(list)

    for i in tqdm(range(len(face_list)), desc="Processing Faces"):
        path_i, _, is_detected_i = face_list[i]

        # Skip if face is not detected in the image
        if not is_detected_i:
            continue

        # Add the current image path to the list of images for the current face
        face_images[i].append(path_i)

        # Iterate over the remaining images to compare with the current image
        for j in range(i + 1, len(face_list)):
            if i != j:
                path_j, _, is_detected_j = face_list[j]

                # Skip if face is not detected in the image
                if not is_detected_j:
                    continue

                # Verify if the faces in the two images are the same
                result = DeepFace.verify(
                    path_i, path_j, model_name=models[1], enforce_detection=False)
                is_same = result['verified']

                # If faces are the same, add the image path to the list of images for the current face
                if is_same:
                    face_images[i].append(path_j)
                    face_list[j][2] = True  # Mark the image as processed

    return face_images


def openCV_face_detect(folder_path):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    images = [os.path.join(folder_path, file) for file in os.listdir(
        folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))]
    if not os.path.exists(unique_face_folder):
        os.makedirs(unique_face_folder)
    i = 0
    for image in images:
        # Load the image
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=10, minSize=(64, 64))

        # Save detected faces
        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]  # Crop the detected face region
            cv2.imwrite(f"{unique_face_folder}/Face_{i}.jpeg", face_img)
            i += 1


def segregate_images_by_person(folder_path, frame_path="temp.png"):
    # Read the input image
    image = cv2.imread(frame_path)
    image = cv2.resize(image, (1024, 1024))
    # Extract faces from the input image
    faces = DeepFace.extract_faces(
        image, detector_backend='opencv', enforce_detection=False)

    if len(faces) != 1:
        print("Error: Expected exactly one face in the input image")
        return []

    main_face = faces[0]["face"]
    # cv2.imshow("A", main_face)
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()  # Close the window after display

    # Save the main face image
    success = cv2.imwrite("main_face.jpeg", main_face)
    if success:
        print("Main face image saved successfully.")
    else:
        print("Error: Failed to save the main face image.")

    # Get list of image files in the specified folder
    images = [os.path.join(folder_path, file) for file in os.listdir(
        folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))]

    # Create a list to store paths of images containing the same face
    all_images = []

    # Iterate through each image in the folder
    for image_path in tqdm(images, desc="Processing Images"):
        # Read the current image
        current_image = cv2.imread(image_path)
        current_image = cv2.resize(current_image, (1024, 1024))

        # Extract faces from the current image
        detected_faces = DeepFace.extract_faces(
            current_image, detector_backend='opencv', enforce_detection=False)

        # Check if any face is detected in the current image
        for face_data in detected_faces:
            # Extract the detected face from the current image
            detected_face = face_data['face']

            # Verify if the detected face matches the main face
            result = DeepFace.verify(
                detected_face, main_face, models[0], enforce_detection=False)
            is_same1 = result['verified']
            # is_same2, simiaritry = verify_faces(
            #     detected_face, main_face, resnet_model)

            if is_same1:
                all_images.append(image_path)
                # cv2.imshow("Main Face", detected_face)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # print("Image:", image_path)
                # print("Distance1:", result['distance'], is_same1)
                # print("Distance2:", simiaritry, is_same2)
                break

    return all_images


# Call the function to recognize faces in real-time
input_folder = "./input/friends"
output_folder = "Output"
result_folder = "Result"
unique_face_folder = f"{output_folder}/Faces"
2
#  For Single Frame Image Segregations
capture_frame()
images = segregate_images_by_person(input_folder,)
print(images)
for j in images:
    copy_images(j, f"{result_folder}")

# For All Faces Images Segregation in a Folder. Please Uncomment Below Code

# if os.path.exists(output_folder):
#     shutil.rmtree(output_folder)
# # openCV_face_detect(input_folder)
# face_list = separate_faces_in_image(input_folde//home/subhan/Downloads/temp.png)
# face_images = consolidate_faces_to_images(face_list)
# for i in face_images:
#     destination_folder = f"{output_folder}/Face-{i}-Images"
#     for j in face_images[i]:
#         copy_images(j, destination_folder)
#     copy_images(
#         f"{unique_face_folder}/face_{i}.jpg", destination_folder)
