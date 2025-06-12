import os
from openai import OpenAI
from google.cloud import storage
import PyPDF2
from io import BytesIO
import logging
from itertools import chain
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv  # Import the dotenv module

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()  # This will load the .env file in your project directory

# Retrieve OpenAI API Key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")  # Use the key stored in .env

# Check if the API key is loaded correctly
if not openai_api_key:
    raise ValueError("OpenAI API key is not set. Please set it in the .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Initialize the text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Function to generate description, keywords, and tags using OpenAI API
def generate_openai_output(text):
    prompt = f"""Please read the following text and provide the response in JSON format:
    {{
        "description": "10-15 word description of the content",
        "keywords": ["5 important keywords"],
        "tags": ["relevant tags such as product name, purpose"]
    }}

    {text}"""

    # Generate the output using OpenAI's GPT-4 model
    response = client.chat.completions.create(
        model="gpt-4",  # Use GPT-4 model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # Lower temperature for more deterministic output
    )
    
    # Correctly handle the OpenAI response
    content = response.choices[0].message.content

    
    
    
    
    # Extract the description, keywords, and tags from the response
   
    print (content)
    return content

# Function to read PDFs from GCS and extract text
def read_pdf_from_gcs(bucket_name, blob_names):
    """Read PDFs from GCS and extract text with error handling"""
    complete_document = []
    print(f"Processing the following blobs: {blob_names}")
    
    # Initialize the storage client once
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for blob_name in blob_names:
        try:
            logger.info(f"Processing blob: {blob_name}")
            blob = bucket.blob(blob_name)
            pdf_bytes = blob.download_as_bytes()
            pdf_file = BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            data = []
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    data.append(text)
                else:
                    logger.warning(f"No text extracted from page {page_num} in blob {blob_name}")
            
            pdf_text = '. '.join(data)

            # Generate Description, Keywords, and Tags using OpenAI
            file_des= generate_openai_output(pdf_text)

            description= file_des.get('description')
            keywords=file_des.get('keywords')
            tags=file_des.get('tags')

            # Print the generated description, keywords, and tags
            print(f"Generated description for {blob_name}: {description}")
            print(f"Generated keywords for {blob_name}: {keywords}")
            print(f"Generated tags for {blob_name}: {tags}")

            logger.info(f"Generated description: {description}")
            logger.info(f"Generated keywords: {keywords}")
            logger.info(f"Generated tags: {tags}")

        except Exception as e:
            logger.error(f"Error processing blob {blob_name}: {e}")
    
    # Return the processed documents
    flattened_docs = list(chain.from_iterable(complete_document))
    return flattened_docs

# Function to process each file content
def embeddings_from_gcs(bucket_name, blob_names):
    """Load PDF documents from GCS and return them as documents"""
    try:
        docs = read_pdf_from_gcs(bucket_name, blob_names)
        if not docs:
            logger.warning("No documents were extracted from the PDFs.")
            return "No documents extracted."

        # Return the documents
        return docs

    except Exception as e:
        logger.error(f"Failed to process GCS files: {e}")
        return f"Error processing GCS files: {e}"

# Main function to run the process
def main():
    """Main function to run the process"""
    bucket_name = "pt-product-1"  # Replace with your GCS bucket name
    blob_names = ["TYTAN DATA SHEET.pdf"]  # Replace with the list of blob names (PDFs) you want to process

    # Call the function to process PDFs from GCS and get documents
    result = embeddings_from_gcs(bucket_name, blob_names)
    
    if isinstance(result, str):
        print(result)  # If the result is a message (e.g., error or warning), print it
    else:
        print("Documents extracted successfully:")
        print(result)  # Print the extracted documents

if __name__ == "__main__":
    main()
