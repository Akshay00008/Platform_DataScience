import os
from dotenv import load_dotenv
import pymongo
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI and Embedding Setup
client = OpenAI(api_key=openai_api_key)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Load FAISS
faiss_path = r"/home/bramhesh_srivastav/Platform_DataScience/website_faiss_index"

vectorstore = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)

# MongoDB connection
mongo_client = pymongo.MongoClient("mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")
db = mongo_client["ChatbotDB"]
guidance_collection = db["guidanceflows"]

# Fetch content from vector store
def fetch_vector_content(query="overview", k=25):
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

# Generate structured guidance using GPT-4o
def generate_guidance(content):
    prompt = f"""
You are a company assistant bot. Based on the content provided below, generate structured business guidance in sections.

Each section should include a clear **section title** (e.g., "Company Overview", "Product Guidance", "Support Services") followed by a short explanation.

--- Content ---
{content}
----------------

Format:
Section Title 1
Explanation...

Section Title 2
Explanation...

(Use double line breaks to separate each section)
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

    for section in sections:
        lines = section.strip().split("\n")
        if len(lines) < 2:
            continue
        title = lines[0].strip()
        explanation = "\n".join(lines[1:]).strip()
        parsed.append({
            "chatbot_id": chatbot_id,
            "version_id": version_id,
            "section_title": title,
            "content": explanation,
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
