from google.cloud import storage
import PyPDF2
from io import BytesIO
import logging
from itertools import chain

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

            # Split the document into chunks
            try:
                docs = text_splitter.create_documents([pdf_text])
                complete_document.append(docs)
            except Exception as e:
                logger.error(f"Error during text splitting for {blob_name}: {e}")

        except Exception as e:
            logger.error(f"Error processing blob {blob_name}: {e}")
    
    # Return the processed documents
    flattened_docs = list(chain.from_iterable(complete_document))
    return flattened_docs

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

def main():
    """Main function to run the process"""
    bucket_name = "your_bucket_name"  # Replace with your GCS bucket name
    blob_names = ["pt-product-1/1748540260489-TYTAN DATA SHEET.pdf"]  # Replace with the list of blob names (PDFs) you want to process

    # Call the function to process PDFs from GCS and get documents
    result = embeddings_from_gcs(bucket_name, blob_names)
    
    if isinstance(result, str):
        print(result)  # If the result is a message (e.g., error or warning), print it
    else:
        print("Documents extracted successfully:")
        print(result)  # Print the extracted documents

if __name__ == "__main__":
    main()
