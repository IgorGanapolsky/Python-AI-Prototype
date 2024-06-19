import concurrent
import json
import os
import re
import time
from pathlib import Path

import PyPDF2
import logging
import openai
from PIL import Image
from rich.console import Console
from rich.text import Text
from tqdm import tqdm

from OpenAI.utils import send_image_to_openai, PDF_DIRECTORY, IMAGES_DIRECTORY, PARSED_PDF_JSON_DIRECTORY


def process_all_docs_into_json():
    docs = []

    pdf_directory_path = Path(PDF_DIRECTORY)
    print("pdf_directory: ", pdf_directory_path)
    all_pdfs = [f for f in pdf_directory_path.iterdir() if f.is_file()]
    print("all_pdfs: ", all_pdfs)
    files = [item for item in all_pdfs if item.is_file()]
    print("files: ", files)

    for f in files:
        print("processing doc " + str(f))
        doc = {
            "filename": f
        }
        path = str(doc["filename"])
        text = extract_text_from_doc(path)
        doc['text'] = text

        # Get the filename without the extension
        filename_without_ext = os.path.splitext(os.path.basename(f))[0]
        # Get all the images for this file
        img_dir = os.path.join(IMAGES_DIRECTORY, filename_without_ext)
        imgs = get_images_from_directory(img_dir)

        pages_description = []

        print(f"Analyzing pages for doc {f}")

        try:
            # Concurrent execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:

                futures = [
                    executor.submit(analyze_image, img_path)
                    for img_path in imgs[1:]
                ]

                with tqdm(total=len(imgs) - 1) as pbar:
                    for _ in concurrent.futures.as_completed(futures):
                        pbar.update(1)

                for future in futures:
                    res = future.result()
                    pages_description.append(res)

            doc['pages_description'] = pages_description
            docs.append(f)
        except KeyboardInterrupt:
            print("Process interrupted by user. Exiting gracefully.")
        finally:
            print("Process completed.")
            executor.shutdown(wait=False)

        # Saving result to json file for later
        json_path = PARSED_PDF_JSON_DIRECTORY
        with open(json_path, 'w') as saved_json:
            json.dump(docs, saved_json)


def remove_citations(text):
    # The pattern matches strings like "Liu et al. (2023a)" and "Zheng et al. (2023)"
    pattern = r"\b\w+\s+et al\.\s+\(\d{4}[a-z]?\)"
    # Replace the matched patterns with an empty string
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text


def pretty_print_chat_message(content):
    content_text = Text("\n" + content, style="green")
    content_text.highlight_regex(r"\b(error|warning)\b", "bold red")

    cleaned_content = remove_citations(str(content_text))

    # Create a Console instance
    console = Console(width=200)
    console.print(cleaned_content)


def print_json_output():
    with open(PARSED_PDF_JSON_DIRECTORY, 'r') as json_file:
        data = json.load(json_file)
        pretty_print_chat_message(json.dumps(data, indent=4))


if __name__ == "__main__":
    logger = logging.getLogger("PyPDF2")
    logger.setLevel(logging.ERROR)
    process_all_docs_into_json()
    print_json_output()
