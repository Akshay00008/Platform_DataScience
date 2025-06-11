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

# MongoDB setup
mongo_client = pymongo.MongoClient("mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")
db = mongo_client["ChatbotDB"]
collection = db['handoffscenarios']

# Vector DB setup
faiss_path = r"/home/bramhesh_srivastav/Platform_DataScience/website_faiss_index"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# OpenAI client
client = OpenAI(api_key=openai_api_key)

# Function to load FAISS index fresh every time
def load_faiss_index():
    """
    Load the FAISS index fresh from disk each time it's called.
    """
    return FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)

# Function to fetch content from FAISS
def search_vector_context(query, k=30):
    """
    Fetch the vector content by performing a similarity search with a fresh FAISS index.
    """
    vectorstore = load_faiss_index()  # Reload FAISS index each time
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

def generate_handoff_guidance(query, chatbot_id, version_id):
    context = search_vector_context(query)

    prompt = f"""
Use the following website content to explore the chatbot's knowledge base.

Generate a set of clear and concise guidelines for when the chatbot should hand off a conversation to a customer care representative. The guidelines should cover:

1. Complex Issues Beyond Chatbot's Scope (account/billing/technical issues)
2. Specific Product or Service Support
3. Escalation Requests (user demands human assistance)
4. Missing Content on Website or YouTube
5. Inquiries Needing Deep Explanation from Website/YouTube

Context:
{context}

Provide output in structured guidance points with section titles.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    guidance_text = response.choices[0].message.content

    # Split the guidance text by sections
    sections = guidance_text.split("\n\n")  # Split by double new lines between sections

    guidance_entries = []

    # Iterate over each section and save it as a separate entry in MongoDB
    for idx, section in enumerate(sections, start=1):
        # Create a separate document for each section
        guidance_entries.append({
            "chatbot_id": ObjectId(chatbot_id),
            "version_id": ObjectId(version_id),
            "section_id": idx,  # Section number (1, 2, 3, etc.)
            "section_title": f"Guideline {idx}",
            "description": section.strip(),
            "category_name": "New",
            "source_type": "ai",
            "is_enabled": False
        })

    # Insert each guideline as a separate document in MongoDB
    if guidance_entries:
        collection.insert_many(guidance_entries)  # Insert all guidelines at once
        print(f"Inserted {len(guidance_entries)} guidelines into MongoDB.")

    return guidance_text
