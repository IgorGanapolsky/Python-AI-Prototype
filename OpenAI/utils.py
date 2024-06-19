import base64
import os
import time
import PyPDF2
import openai

from io import BytesIO
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

# Load the .env file
load_dotenv()
# Initialize the OpenAI client (no need to pass OPENAI_API_KEY directly)
openai.api_key = os.getenv('OPENAI_API_KEY')

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIRECTORY = BASE_DIR / "data/example_pdfs"
IMAGES_DIRECTORY = BASE_DIR / "data/processed_images"
PARSED_PDF_JSON_DIRECTORY = BASE_DIR / "data/parsed_pdfs_json"

system_prompt = '''
You are an expert analyzer of HuggingFace technical papers which were converted from PDF to  image format.  
Your task is to describe the content of the image as if you were delivering a presentation about it. Here are some 
specific guidelines to follow:

- If there are diagrams, please describe what they depict and explain their meaning. For example, if there is a process flow 
diagram, walk through the steps like "The process begins with X, then proceeds to step Y, followed by Z" etc.

- If there are data tables, please summarize the information in them logically. For instance, if there is a pricing table, you 
could say something like "The prices listed are: $X for Item A, $Y for Item B, and $Z for Item C."

- Please focus on describing the actual content and information being conveyed, rather than referring to the format like "the 
diagram shows" or "the text says". Imagine your audience cannot see the image, so be thorough in explaining all the 
relevant content.

DO NOT talk about terms referring to the content format.
DO NOT talk about the content type in your output.
DO NOT talk about References in your output.
DO NOT talk about 'http' or 'https' url links of any kind in your output.

For example: if there is a diagram/chart and text on the image, talk about both without mentioning that one is a chart 
and the other is text. Simply describe what you see in the diagram and what you understand from the text.

- Be concise in your wording, but make sure to cover all the key information, as if your listener needs to fully 
understand the content without seeing the image. Do not leave out any important details.

- Use proper grammar and complete sentences. Do not trail off without finishing your thoughts. Punctuate correctly, 
capitalize appropriately, and avoid run-on sentences.

- Skip over irrelevant elements like page numbers, headers and footers, section numbers, URLs, etc. Only describe the 
main content.

After examining the image, provide your content description in the following format:

If there is a clear title, list the title followed by your description, like this:

<title>The Title</title>

<content>
Your detailed content description goes here, following the guidelines above. Use multiple paragraphs if needed to 
organize the information.
</content>

If there is no obvious title, just provide your content description inside <content> tags.

------

Remember, the goal is to convey the key information as if you were explaining it to an audience who cannot see the 
image. Be clear, logical, concise and complete in your description.

'''


# Convert image to base64 encoded in a data URI format.
def get_img_uri(image):
    buffer = BytesIO()
    image.save(buffer, format="png")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    uri = f"data:image/png;base64,{base64_image}"
    return uri


# Analyze the image by sending it to OpenAI Chat API.
def send_image_to_openai(image):
    img_uri = get_img_uri(image)

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_uri
                        }
                    },
                ],
            }
        ],
        max_tokens=600,
        top_p=0.1
    )

    return response.choices[0].message.content


def analyze_image(img_path):
    print("analyzing " + str(img_path))
    page_data = None
    try:
        img = Image.open(img_path)
        page_data = send_image_to_openai(img)
    except openai.RateLimitError as e:
        # Extract the number of seconds to wait from the error message
        match = re.search(r'Please try again in (\d+\.\d+)s.', str(e))
        if match:
            wait_time = float(match.group(1))
            print(f"Rate limit exceeded. Waiting for {wait_time} seconds.")
            time.sleep(wait_time)
            # Retry the API call
            page_data = analyze_image(img_path)

    print("data: ", page_data)
    return page_data


def extract_text_from_doc(path):
    pdf = open(path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf)
    total_pages = len(pdf_reader.pages)
    for page in range(total_pages):
        page_content = pdf_reader.pages[page]
        extracted_text = page_content.extract_text()
        page_text = [extracted_text]
    return page_text


def get_images_from_directory(img_directory_path):
    image_file_paths = [os.path.join(img_directory_path, f) for f in os.listdir(img_directory_path)]
    return image_file_paths


def save_images(page_images, file_name):
    for i, img in enumerate(page_images):
        img.save(f"{IMAGES_DIRECTORY}/{file_name}/image_{i}.png")
