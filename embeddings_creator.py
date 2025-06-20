from typing import List
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
from utility.logger_file import Logs
import json
import numpy as np
from dotenv import load_dotenv
 
app = FastAPI()
logger = Logs()  # Instantiate the logger here
 
try:
 
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error('Open AI key is missing')

    load_dotenv()    
    api = os.getenv('OPENAI_API_KEY')
    model = os.getenv('GPT_model')
    model_provider = os.getenv('GPT_model_provider')    
 
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
 
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
 
    text_splitter = SemanticChunker(OpenAIEmbeddings(), number_of_chunks=1000)
 
except Exception as e:
    logger.error(f"Reconciliation failed: {str(e)}")
    raise ValueError(f"Reconciliation failed: {str(e)}")
 
def document_creator(text: str):
    try:
        long_doc = [Document(page_content=text)]
        docs = text_splitter.split_documents(long_doc)
        return docs
    except Exception as e:
        logger.error(f"Document creator failed: {str(e)}")
        raise
 
def read_pdf_from_gcs(bucket_name, blob_names):
    """Read PDFs from GCS and extract text with error handling"""
    try:
        complete_document = []
        logger.info(f"Processing blobs: {', '.join(blob_names)}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
 
        for blob_name in blob_names:
            blob = bucket.blob(blob_name)
            if not blob.exists():
                logger.warning(f"⚠️ Blob '{blob_name}' not found in bucket '{bucket_name}'. Skipping.")
                raise Exception(f"Blob '{blob_name}' not found in bucket '{bucket_name}'")
 
            pdf_bytes = blob.download_as_bytes()
            pdf_file = BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            full_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            docs = document_creator(full_text)
            complete_document.append(docs)
 
        return chain.from_iterable(complete_document)
 
    except Exception as e:
        logger.error(f"PDF reader from GCS failed: {e}")
        raise
 




# Embedding function (assuming OpenAIEmbeddings is already initialized globally or passed as a parameter)
embeddings = OpenAIEmbeddings(model="text-embedding-003")  # Example embedding initialization

def embeddings_from_gcb(bucket_name, blob_names):
    try:
        # Assuming `read_pdf_from_gcs` is a function that reads PDFs from Google Cloud Storage
        docs = read_pdf_from_gcs(bucket_name, blob_names)
        
        if not docs:
            logger.warning("No documents were extracted from the PDFs.")
            return "No documents extracted."

        # Check if the FAISS index exists
        if os.path.exists("faiss_index"):
            try:
                # Load the existing FAISS index
                vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                logger.info("Loaded existing FAISS index.")

                # Generate embeddings for the new documents
                logger.info("Generating embeddings for the new documents...")
                new_doc_embeddings = embeddings.embed_documents(docs)

                # Ensure the embeddings are numpy arrays (required by FAISS)
                new_doc_embeddings = np.array(new_doc_embeddings)
                logger.info(f"Generated embeddings for {len(docs)} new documents.")

                # Create a new FAISS index for the new documents
                dim = len(new_doc_embeddings[0])  # Get the dimensionality of the embeddings
                new_index = faiss.IndexFlatL2(dim)  # Create a new index with L2 distance metric
                new_index.add(new_doc_embeddings)  # Add the new embeddings to the new index

                # Merge the existing FAISS index with the new index
                vector_store.index.merge(new_index)
                logger.info("Merged the existing FAISS index with the new documents.")

                # Save the updated FAISS index
                vector_store.save_local("faiss_index")
                logger.info("Documents added and FAISS index saved successfully.")

                return "FAISS index updated successfully!"

            except Exception as e:
                logger.error(f"Failed to add documents to FAISS or save: {e}")
                return f"Error adding documents or saving FAISS index: {e}"

        else:
            try:
                # Create a new FAISS index for the new documents
                logger.info("Creating a new FAISS index for the new documents...")
                new_doc_embeddings = embeddings.embed_documents(docs)
                new_doc_embeddings = np.array(new_doc_embeddings)

                # Create a new FAISS index
                dim = len(new_doc_embeddings[0])  # Get the dimensionality of the embeddings
                index = faiss.IndexFlatL2(dim)  # Create a new index with L2 distance metric
                index.add(new_doc_embeddings)  # Add the new embeddings to the index

                # Create the vector store with the new index
                vector_store = FAISS(
                    embedding_function=embeddings,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )

                # Save the newly created FAISS index
                vector_store.save_local("faiss_index")
                logger.info("Created new FAISS index and saved it.")
                return "Created and saved new FAISS index."

            except Exception as e:
                logger.error(f"Failed to create new FAISS index: {e}")
                return f"Error creating FAISS index: {e}"

    except Exception as e:
        logger.error(f"Failed to process documents or create embeddings: {e}")
        return f"Error processing documents or creating embeddings: {e}"


 
def embeddings_from_website_content(json_data):
    try:
        # Process website content here
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
        faiss_index.save_local("website_faiss_index")
 
        logger.info(f"✅ FAISS index saved with {len(text_chunks)} chunks.")
        return "FAISS vector store saved successfully!"
 
    except Exception as e:
        logger.error(f"Error in embeddings_from_website_content: {e}")
        return f"An error occurred: {e}"
