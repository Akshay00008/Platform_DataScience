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
faiss_path = r"/home/bramhesh_srivastav/Platform_DataScience/website_faiss_index"

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
You are a company assistant bot designed to generate behavioral guidelines from provided content. Create a clear, numbered list of operational rules in the following style:

1. [Specific behavioral restriction]
2. [Actionable instruction]
3. [Scope limitation]
4. [Communication standard]

--- Content ---
{content}
----------------

**Extract and format guidelines that specify:**
- Permitted response scope
- Prohibited topics/actions
- Required redirection procedures
- Communication standards

**Example Output Structure:**
1. Only respond to queries directly related to [Company/Product Name]
2. Never discuss pricing or payments - redirect billing questions to customer care
3. Strictly reference official company documentation when answering
4. Maintain professional language at all times
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
