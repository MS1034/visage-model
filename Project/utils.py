import os
import shutil


def copy_images(imagePath, destination_folder):
    """
    Creates a copy of the image located at the specified path and saves it to the destination folder.

    Args:
        imagePath (str): The path to the source image.
        destination_folder (str): The path to the destination folder where the copied image will be saved.

    Returns:
        None
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    image_filename = os.path.basename(imagePath)

    destination_path = os.path.join(destination_folder, image_filename)

    shutil.copy(imagePath, destination_path)


def images_in_folder(folder_path):
    """
    Retrieves a list of paths to image files within a specified folder.

    Args:
        folder_path (str): The path to the folder containing the image files.

    Returns:
        list: A list of paths to image files (JPEG, PNG, or JPG) within the specified folder.
    """
    # Generate a list of file paths within the folder that end with '.jpg', '.png', or '.jpeg'
    images = [os.path.join(folder_path, file) for file in os.listdir(
        folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))]

    return images
