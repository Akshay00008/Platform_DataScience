# from typing import List
# from google.cloud import storage
# import os
# from langchain_core.documents import Document
# from io import BytesIO
# import PyPDF2
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from itertools import chain
# import faiss
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_community.vectorstores import FAISS
# import getpass
# from utility.logger_file import Logs
# from fastapi import FastAPI
# import logging
# import json
 
# app = FastAPI()
# logger=Logs()
 
 
# try:
 
#     if not os.environ.get("OPENAI_API_KEY"):
#         logger.error('Open AI key is missing')
 
#     if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
#         logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
 
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
 
#     text_splitter = SemanticChunker(OpenAIEmbeddings(), number_of_chunks=1000)
 
 
# except Exception as e:
 
#     logger.error(f"Initialization failed: {e}")
#     raise
 
 
 
 
# def document_creator(text:str):
#     try:    
 
#         long_doc = [Document(page_content=text)]
#         docs=text_splitter.split_documents(long_doc)
#         return docs
   
#     except Exception as e:
#         logger.error(f"document creator {str(e)}")
#         raise
 
 
 
# def read_pdf_from_gcs(bucket_name, blob_names):
#     """Read PDFs from GCS and extract text with error handling"""
#     try:
#         complete_document = []
        
#         storage_client = storage.Client()
#         bucket = storage_client.bucket(bucket_name)
 
#         for blob_name in blob_names:
#             logger.info(f"Processing blob: {blob_name}")
           
#             blob = bucket.blob(blob_name)
#             if not blob.exists():
#                 logger.warning(f"⚠️ Blob '{blob_name}' not found in bucket '{bucket_name}'. Skipping.")
#                 raise Exception(f"Blob '{blob_name}' not found in bucket '{bucket_name}'")
 
 
#             pdf_bytes = blob.download_as_bytes()
#             pdf_file = BytesIO(pdf_bytes)
#             pdf_reader = PyPDF2.PdfReader(pdf_file)
#             full_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
#             docs=document_creator(full_text)
#             complete_document.append(docs)
 
                   
       
#         return chain.from_iterable(complete_document)
   
#     except Exception as e:
#         logger.error(f"pdf reader form gcs failed: {e}")
#         raise
       
 
 
 
# def embeddings_from_gcb(bucket_name, blob_names):
#     try:
#         docs = read_pdf_from_gcs(bucket_name, blob_names)

#         if not docs:
#             logger.warning("No documents were extracted from the PDFs.")
#             return "No documents extracted."

#         # Check if FAISS index already exists
#         if os.path.exists("faiss_index"):
#             try:
#                 vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#                 logger.info("Loaded existing FAISS index.")

#                 # 🔄 Add new documents
#                 vector_store.add_documents(docs)
#                 logger.info("Added new documents to existing FAISS index.")
                
#                 result_message = "Updated existing FAISS index with new documents"
#             except Exception as e:
#                 logger.error(f"Failed to load or update existing FAISS index: {e}")
#                 return f"Error updating existing FAISS index: {e}"
#         else:
#             try:
#                 # Create a new index
#                 dim = len(embeddings.embed_query(docs[0].page_content))
#                 index = faiss.IndexFlatL2(dim)
#                 vector_store = FAISS(
#                     embedding_function=embeddings,
#                     index=index,
#                     docstore=InMemoryDocstore(),
#                     index_to_docstore_id={},
#                 )

#                 vector_store.add_documents(docs)
#                 logger.info("Created new FAISS index and added documents.")
                
#                 result_message = "Created new FAISS index"
#             except Exception as e:
#                 logger.error(f"Failed to create FAISS index: {e}")
#                 return f"Error creating new FAISS index: {e}"

#         # 💾 Save updated/new index
#         vector_store.save_local("faiss_index")
#         logger.info("Saved FAISS index.")

#         return result_message

#     except Exception as e:
#         logger.error(f"Embeddings from GCB failed: {e}")
#         return f"Embeddings processing failed: {e}"
 
# def embeddings_from_website_content(json_data):
   
 
#     # Load JSON data
   
 
#     documents = []
#     metadata = []
 
#     for idx, item in enumerate(json_data):
#         text_parts = []
#         if item.get("Title"):
#             text_parts.append(item["Title"])
#         if item.get("Meta Description") and item["Meta Description"] != "No description":
#             text_parts.append(item["Meta Description"])
#         if item.get("Headings"):
#             for key, values in item["Headings"].items():
#                 text_parts.extend(values)
#         if item.get("Paragraphs"):
#             text_parts.extend(item["Paragraphs"])
 
#         combined_text = " ".join(text_parts).strip()
#         if combined_text:
#             documents.append(combined_text)
#             metadata.append({"source": f"web_doc_{idx}"})
 
#     if not documents:
#         raise ValueError("No valid website content found for embedding.")
 
#     # Split text into chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=20,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     text_chunks = []
#     chunk_metadata = []
 
#     for i, doc in enumerate(documents):
#         chunks = text_splitter.split_text(doc)
#         text_chunks.extend(chunks)
#         chunk_metadata.extend([metadata[i]] * len(chunks))
 
#     if not text_chunks:
#         raise ValueError("No text chunks were created from website content.")
 
#     # Initialize OpenAI embeddings
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
 
#     # Build FAISS index
#     faiss_index = FAISS.from_texts(text_chunks, embeddings, metadatas=chunk_metadata)
 
#     # Save index and metadata
#     faiss_index.save_local("website_faiss_index")
 
#     print(f"✅ FAISS index saved with {len(text_chunks)} chunks.")
#     return "FAISS vector store saved successfully!"




from google.cloud import storage
import os
from langchain_core.documents import Document
from io import BytesIO
import PyPDF2
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from itertools import chain
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import getpass
from fastapi import FastAPI
import logging
from typing import List
from utility.logger_file import Logs

# Logger setup (assuming the logger is defined in your project)
logger = logging.getLogger("faiss_index_logger")

# Ensure the environment variables are set
try:
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error('Open AI key is missing')

    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    text_splitter = SemanticChunker(OpenAIEmbeddings(), number_of_chunks=1000)

except Exception as e:
    logger.error(f"Initialization failed: {e}")
    raise ValueError(f"Initialization failed: {e}")


def document_splitter(text: str) -> List:
    # Assuming you have a text splitting logic, which uses SemanticChunker
    try:
        long_doc = [Document(page_content=text)]
        # Use your document splitter here
        docs = text_splitter.split_documents(long_doc)
        return docs
    except Exception as e:
        logger.error(f"Document creator failed: {str(e)}")
        raise


def read_pdf_from_gcs(bucket_name: str, blob_names: List[str]) -> List[str]:
    """Read PDFs from GCS and extract text with error handling"""
    try:
        complete_document = []

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        for blob_name in blob_names:
            logger.info(f"Processing blob: {blob_name}")
            blob = bucket.blob(blob_name)
            if not blob.exists():
                logger.warning(f"⚠️ Blob '{blob_name}' not found in bucket '{bucket_name}'. Skipping.")
                continue

            pdf_bytes = blob.download_as_bytes()
            pdf_file = BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            full_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            docs = document_splitter(full_text)
            complete_document.append(docs)

        return list(chain.from_iterable(complete_document))

    except Exception as e:
        logger.error(f"PDF reading failed: {str(e)}")
        raise


def load_or_create_faiss_index(docs: List[str], chatbot_id: str, version_id: str) -> str:
    try:
        index_dir = "/home/bramhesh_srivastav/Platform_DataScience/faiss_indexes"  # Directory to store FAISS indexes
        
        os.makedirs(index_dir, exist_ok=True)

        # Use chatbot_id and version_id to create a unique index filename
        index_filename = f"{chatbot_id}_{version_id}_faiss_index"
        index_path = os.path.join(index_dir, index_filename)

        if os.path.exists(index_path):
            print('Using existing FAISS index')
            vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            logger.info("Loaded existing FAISS index.")
        else:
            print('Creating new FAISS index')
            dim = len(embeddings.embed_query("hello world"))  # Get the dimensionality of the embeddings
            index = faiss.IndexFlatL2(dim)  # FAISS index for L2 distance
            vector_store = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            logger.info("Created new FAISS index.")

        # Add documents to the FAISS index
        vector_store.add_documents(documents=docs)

        # Save the FAISS index to the file
        vector_store.save_local(index_path)
        logger.info(f"FAISS index saved as: {index_path}")

        return "Successfully created or loaded FAISS index"

    except Exception as e:
        logger.error(f"Failed to create FAISS index: {str(e)}")
        return f"Failed to create FAISS index: {str(e)}"


def embeddings_from_gcb(chatbot_id: str, version_id: str, bucket_name: str, blob_names: List[str]) -> str:
    try:
        # Retrieve documents from Google Cloud Storage
        docs = read_pdf_from_gcs(bucket_name, blob_names)
        logger.info(f"Documents extracted: {len(docs)}")

        if not docs:
            logger.warning("No documents were extracted from the PDFs.")
            return "No documents extracted."

        # Load or create the FAISS index
        reply = load_or_create_faiss_index(docs=docs, chatbot_id=chatbot_id, version_id=version_id)
        return reply

    except Exception as e:
        logger.error(f"Error in embeddings_from_gcb: {e}")
        return f"An error occurred: {e}"

   
 
 
 
 
 
def embeddings_from_website_content(json_data,  chatbot_id, version_id):
   
    documents = []
    metadata = []
 
    for idx, item in enumerate(json_data):
        text_parts = []
        if item.get("Title"):
            text_parts.append(item["Title"])
        if item.get("Meta Description") and item["Meta Description"] != "No description":
            text_parts.append(item["Meta Description"])
        if item.get("Headings"):
            for key, values in item["Headings"].items():
                text_parts.extend(values)
        if item.get("Paragraphs"):
            text_parts.extend(item["Paragraphs"])
 
        combined_text = " ".join(text_parts).strip()
        if combined_text:
            documents.append(combined_text)
            metadata.append({"source": f"web_doc_{idx}"})
 
    if not documents:
        raise ValueError("No valid website content found for embedding.")
 
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    text_chunks = []
    chunk_metadata = []
 
    for i, doc in enumerate(documents):
        chunks = text_splitter.split_text(doc)
        text_chunks.extend(chunks)
        chunk_metadata.extend([metadata[i]] * len(chunks))
 
    if not text_chunks:
        raise ValueError("No text chunks were created from website content.")
 
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
 
    # Build FAISS index
    faiss_index = FAISS.from_texts(text_chunks, embeddings, metadatas=chunk_metadata)
 
    # Save index and metadata
    index_dir  = "/home/bramhesh_srivastav/Platform_DataScience/faiss_indexes"  # Directory where indexes will be saved
    os.makedirs(index_dir, exist_ok=True)
    # Generate a unique index filename based on chatbot_id and version_id
    index_filename = f"{chatbot_id}_{version_id}_faiss_index_website"

    # Save the index and metadata to a file
    faiss_index.save_local(os.path.join(index_dir, index_filename))
    print(f"FAISS index saved as: {os.path.join(index_dir, index_filename)}")
 
 
    print(f"✅ FAISS index saved with {len(text_chunks)} chunks.")
    return "FAISS vector store saved successfully!"