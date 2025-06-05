# import os
# from dotenv import load_dotenv
# import pymongo
# import re
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from openai import OpenAI

# # Load env variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

# # Initialize OpenAI client
# client = OpenAI(api_key=openai_api_key)

# # Initialize embedding model
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# # Load FAISS vectorstore
# faiss_path = r"C:\Users\hp\Desktop\Platform_16-05-2025\Platform_DataScience\website_faiss_index"
# vectorstore = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)

# # MongoDB setup
# mongo_client = pymongo.MongoClient("mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")
# db = mongo_client["ChatbotDB"]
# collection = db['faqs']

# app = FastAPI()

# class QueryRequest(BaseModel):
#     query: str
#     top_k: int = 20
#     generated_faq_count: int = 50

# def search_faiss(query, k=10):
#     results = vectorstore.similarity_search(query, k=k)
#     return [doc.page_content for doc in results]

# def extract_existing_faqs(chunks):
#     joined_chunks = "\n\n".join(chunks)
#     prompt = f"""
# You are an AI assistant. The following are website or document text chunks.

# Extract any frequently asked questions (FAQs) and their answers if available.

# ---
# {joined_chunks}
# ---

# Return the output as a list of Q&A pairs like:
# Q: ...
# A: ...
# """
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.3,
#     )
#     return response.choices[0].message.content

# def generate_faqs_from_vectors(chunks, target_count=50):
#     joined_chunks = "\n\n".join(chunks[:30])
#     prompt = f"""
# Based on the following content, generate {target_count} relevant and useful Frequently Asked Questions (FAQs) with concise answers.

# ---
# {joined_chunks}
# ---

# Return the output as:
# Q: ...
# A: ...
# """
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.5,
#     )
#     return response.choices[0].message.content

# def parse_faq_text(faq_text):
#     pattern = r"Q:\s*(.+?)\s*A:\s*(.+?)(?=\nQ:|\Z)"
#     matches = re.findall(pattern, faq_text, re.DOTALL)
#     faqs = []
#     for q, a in matches:
#         faqs.append({
#             "question": q.strip(),
#             "answer": a.strip()
#         })
#     return faqs

# def save_faqs_to_mongo(faq_list):
#     if not faq_list:
#         return 0
#     result = collection.insert_many(faq_list)
#     return len(result.inserted_ids)

# @app.post("/faqs")
# async def faqs_endpoint(request: QueryRequest):
#     try:
#         top_chunks = search_faiss(request.query, k=request.top_k)

#         extracted_faq_text = extract_existing_faqs(top_chunks)
#         extracted_faqs = parse_faq_text(extracted_faq_text)
#         inserted_existing_count = save_faqs_to_mongo(extracted_faqs)

#         generated_faq_text = generate_faqs_from_vectors(top_chunks, target_count=request.generated_faq_count)
#         generated_faqs = parse_faq_text(generated_faq_text)
#         inserted_generated_count = save_faqs_to_mongo(generated_faqs)

#         return {
#             "extracted_faq_text": extracted_faq_text,
#             "inserted_existing_faq_count": inserted_existing_count,
#             "generated_faq_text": generated_faq_text,
#             "inserted_generated_faq_count": inserted_generated_count,
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    


import os
from dotenv import load_dotenv
import pymongo
import re

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Load env variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Load FAISS vectorstore
faiss_path = r"/home/bramhesh_srivastav/Platform_DataScience/website_faiss_index"
vectorstore = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)

# MongoDB setup
mongo_client = pymongo.MongoClient("mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")
db = mongo_client["ChatbotDB"]
collection = db['faqs']

def search_faiss(query, k=10):
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

def extract_existing_faqs(chunks):
    joined_chunks = "\n\n".join(chunks)
    prompt = f"""
You are an AI assistant. The following are website or document text chunks.

Extract any frequently asked questions (FAQs) and their answers if available.

---
{joined_chunks}
---

Return the output as a list of Q&A pairs like:
Q: ...
A: ...
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

def generate_faqs_from_vectors(chunks, target_count=50):
    joined_chunks = "\n\n".join(chunks[:30])
    prompt = f"""
Based on the following content, generate {target_count} relevant and useful Frequently Asked Questions (FAQs) with concise answers.

---
{joined_chunks}
---

Return the output as:
Q: ...
A: ...
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content

def parse_faq_text(faq_text):
    pattern = r"Q:\s*(.+?)\s*A:\s*(.+?)(?=\nQ:|\Z)"
    matches = re.findall(pattern, faq_text, re.DOTALL)
    faqs = []
    for q, a in matches:
        faqs.append({
            "question": q.strip(),
            "answer": a.strip()
        })
    return faqs

def save_faqs_to_mongo(faq_list):
    if not faq_list:
        return 0
    result = collection.insert_many(faq_list)
    return len(result.inserted_ids)

