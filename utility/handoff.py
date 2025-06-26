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

# MongoDB Setup
mongo_client = pymongo.MongoClient("mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")
db = mongo_client["ChatbotDB"]
collection = db['handoffscenarios']

# FAISS and Embedding Model Setup
# faiss_path = r"/home/bramhesh_srivastav/Platform_DataScience/website_faiss_index"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# OpenAI Client Setup
client = OpenAI(api_key=openai_api_key)

# Function to Load FAISS Index Fresh Every Time
def load_faiss_index(chatbot_id,version_id):
    """
    Load the FAISS index fresh from disk each time it's called.

    """
    faiss_index_dir = "/home/bramhesh_srivastav/Platform_DataScience/faiss_indexes"
    
    # Create the unique index filename based on chatbot_id and version_id
    # faiss_index_filename = f"{chatbot_id}_{version_id}_faiss_index"
    faiss_index_website = f"{chatbot_id}_{version_id}_faiss_index_website"

    faiss_path = os.path.join(faiss_index_dir, faiss_index_website)

    # faiss_path = r"/home/bramhesh_srivastav/Platform_DataScience/website_faiss_index"
    
    return FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)

# Function to Fetch Content from FAISS
def search_vector_context(chatbot_id,version_id,query, k=30):
    """
    Fetch the vector content by performing a similarity search with a fresh FAISS index.
    """
    vectorstore = load_faiss_index(chatbot_id,version_id)  # Reload FAISS index each time
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

# Function to Generate Handoff Guidance
def generate_handoff_guidance(query, chatbot_id, version_id):
    """
    Generate structured handoff guidance using GPT-4o based on the query and context.
    """
    context = search_vector_context(chatbot_id,version_id,query)

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

Provide output in structured guidance points with section titles donot include ### or numbers any where please.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    guidance_text = response.choices[0].message.content

    # Split the guidance text by sections (double new lines between sections)
    sections = guidance_text.split("\n\n")  # Ensure it splits by sections correctly
    
    # Ensure only 4 guidelines are created
    guidance_entries = []
    for idx, section in enumerate(sections[:5], start=2):  # Only take the first 4 sections
        description = section.strip()  # Remove leading/trailing whitespaces
        
        # Skip appending if description is empty or just spaces
        if description and description != " ":
            # Create a separate document for each section
            guidance_entries.append({
                "chatbot_id": ObjectId(chatbot_id),
                "version_id": ObjectId(version_id),
                "section_title": f"Guideline {idx}",  # Markdown H2 format for titles
                "description": description,  # Regular Markdown content
                "category_name": "New",
                "source_type": "ai",
                "is_enabled": False
            })

    # If guidance_entries is populated, proceed with insertion
    if guidance_entries:
        collection.insert_many(guidance_entries)  # Insert all guidelines at once
        print(f"Inserted {len(guidance_entries)} guidelines into MongoDB.")
