import os
from dotenv import load_dotenv
import pymongo
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from bson import ObjectId

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI and Embedding Setup
client = OpenAI(api_key=openai_api_key)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# FAISS vector store path
faiss_path = r"C:\Users\hp\Desktop\Platform_16-05-2025\Platform_DataScience\website_faiss_index"#/home/bramhesh_srivastav/Platform_DataScience/website_faiss_index"

# MongoDB connection
mongo_client = pymongo.MongoClient("mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")
db = mongo_client["ChatbotDB"]
guidance_collection = db["guidanceflows"]

# Function to load FAISS index fresh every time
def load_faiss_index():
    """
    Load the FAISS index fresh from disk each time it's called.
    """
    return FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)

# Fetch content from vector store
def fetch_vector_content(query="overview", k=25):
    """
    Fetch the vector content by performing a similarity search with a fresh FAISS index.
    """
    vectorstore = load_faiss_index()  # Reload FAISS index each time
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

# Generate structured guidance using GPT-4o
def generate_guidance(content):
    prompt = f"""
You are a company assistant bot designed to generate operational behavioral guidelines from provided content. Your task is to extract and clearly format all relevant behavioral restrictions, action instructions, scope limitations, redirection procedures, and communication standards as a numbered list.



Formatting Rules:



Organize the output into clear section titles, using the following categories (add or adjust as needed):



Response Scope



Prohibited Topics and Actions



Redirection Procedures



Communication Standards



For each section, use sub-numbering for each specific guideline.

(e.g., 1.1, 1.2 under section 1; 2.1, 2.2 under section 2, etc.)



Do not mix explanation numbers across sections. Each section’s guidelines must begin with its section number and sub-number (e.g., 1.1, 2.1, 3.1…).



Extraction Criteria:



Extract and format only the guidelines that specify:



Permitted response scope



Prohibited topics/actions



Required redirection procedures



Communication standards





Example Output Structure:



1. Response Scope

  1.1 Only respond to queries directly related to [Company/Product Name].

  1.2 Do not answer questions unrelated to company offerings.



2. Prohibited Topics and Actions

  2.1 Never discuss pricing or payments.

  2.2 Do not provide legal advice.



3. Redirection Procedures

  3.1 Redirect billing questions to customer care.

  3.2 Forward legal inquiries to the company’s legal department.



4. Communication Standards

  4.1 Maintain professional and respectful language.

  4.2 Reference only official company documentation in responses.



Your task:

Whenever content is provided between "--- Content ---" {content} and "----------------", extract and format the operational behavioral guidelines in the structure above.  
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content

# Parse structured guidance into a list of documents
def parse_guidance(text, chatbot_id, version_id):
    sections = text.strip().split("\n\n")
    parsed = []

    try:
        chatbot_oid = ObjectId(chatbot_id)
        version_oid = ObjectId(version_id)
    except Exception as e:
        print("Invalid ObjectId:", e)
        return 0

    for section in sections:
        lines = section.strip().split("\n")
        if len(lines) < 2:
            continue
        title = lines[0].strip()
        explanation = "\n".join(lines[1:]).strip()
        parsed.append({
            "chatbot_id": chatbot_oid,
            "version_id": version_oid,
            "section_title": title,
            "category_name" : "New",
            "ai_category_name" : "Old",
            "source_type" : "ai",
            "description": explanation,
            "is_enabled": False
        })
    return parsed

# Save to MongoDB
def save_guidance_to_mongo(guidance_docs):
    if not guidance_docs:
        print("No guidance to store.")
        return
    result = guidance_collection.insert_many(guidance_docs)
    print(f"Inserted {len(result.inserted_ids)} guidance sections.")

# Main trigger
def run_guidance_pipeline(chatbot_id, version_id, query="overview"):
    content = fetch_vector_content(query=query)
    structured_text = generate_guidance(content)
    structured_docs = parse_guidance(structured_text, chatbot_id, version_id)
    save_guidance_to_mongo(structured_docs)
    return structured_docs
