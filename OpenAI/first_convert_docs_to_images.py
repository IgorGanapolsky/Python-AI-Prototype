import os

from IPython.display import display, Image
from rich import print
from pdf2image import convert_from_path

from OpenAI.utils import PDF_DIRECTORY, IMAGES_DIRECTORY, save_images


def convert_doc_to_images(pdf_file):
    """
    Convert a PDF document into a series of images.

    This function performs the following steps:
    1. Checks if the images directory exists, and creates it if it does not.
    2. Converts the specified PDF file into a series of images.
    3. Saves the images to a directory named after the PDF file (without extension).
    4. Displays each image.

    Parameters:
    pdf_file (str): The name of the PDF file to be converted.

    Returns:
    None
    """
    # Create the directory if it does not exist
    if not os.path.exists(IMAGES_DIRECTORY):
        os.makedirs(IMAGES_DIRECTORY)

    # Convert each PDF file into a series of images.
    pdf_abs_path = PDF_DIRECTORY / pdf_file

    # Strip the file extension from the file name
    file_name, file_extension = os.path.splitext(pdf_abs_path)
    file_name = os.path.basename(file_name)

    images_dir_path = f"{IMAGES_DIRECTORY}/{file_name}"
    # Create the directory to process images if it does not already exist
    if not os.path.exists(images_dir_path) or not os.listdir(images_dir_path):
        print("converting ", pdf_abs_path, " to images...")
        images = convert_from_path(pdf_abs_path)
        os.makedirs(f"{IMAGES_DIRECTORY}/{file_name}")
        save_images(images, file_name)
        for image in images:
            display(image)
    else:
        print(f"Images for {pdf_file} already exist.")


def iterate_docs():
    """
    Iterate through all PDF files in the specified directory and convert each one into a series of images.

    This function performs the following steps:
    1. Retrieves a list of all PDF files in the PDF_DIRECTORY.
    2. Iterates through each PDF file in the directory.
    3. Opens each PDF file in binary read mode.
    4. Calls the convert_doc_to_images function to convert each PDF into images.

    Parameters:
    None

    Returns:
    None
    """

    pdf_files = os.listdir(PDF_DIRECTORY)

    for pdf_file in pdf_files:
        with open(os.path.join(PDF_DIRECTORY, pdf_file), 'rb') as file:
            # Convert each PDF file into a series of images.
            convert_doc_to_images(pdf_file)


if __name__ == "__main__":
    iterate_docs()
