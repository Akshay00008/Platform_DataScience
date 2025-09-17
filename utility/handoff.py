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
db = mongo_client["ChatbotDB-DEV"]
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
    faiss_index_dir = "/home/bramhesh_srivastav/platformdevelopment/faiss_indexes"
    
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
    Generate structured handoff guidance using GPT-4o based on the query and optional context.
    """
    try:
        # Attempt to retrieve context; may return None or empty string
        context = search_vector_context(chatbot_id, version_id, query)
        
        # If context is available, use context-based prompt
        if context and context.strip():
            prompt = f"""
Use the following website content to explore the chatbot's knowledge base.

--- Content ---
{context}
----------------

Generate a set of clear and concise guidelines for when the chatbot should hand off a conversation to a customer care representative. The guidelines should cover:

Complex Issues Beyond Chatbot's Scope (account/billing/technical issues)
Specific Product or Service Support
Escalation Requests (user demands human assistance)
Missing Content on Website or YouTube
Inquiries Needing Deep Explanation from Website/YouTube

Provide output in structured guidance points with section titles. Do not include ### or numbers anywhere. Use dashes for each item.
"""
        else:
            # No context or empty context
            prompt = """
You are an expert customer-experience designer tasked with creating comprehensive escalation guidelines for a support chatbot when no existing content is available. Generate a set of clear and concise guidelines for when the chatbot should hand off a conversation to a customer care representative. The guidelines should cover:

Complex Issues Beyond Chatbot's Scope (account/billing/technical issues)
Specific Product or Service Support
Escalation Requests (user demands human assistance)
Missing Content or Knowledge Gaps
Inquiries Needing Deep Explanation

Provide output in structured guidance points with section titles. Do not include ### or numbers anywhere. Use dashes for each item.
"""
    except Exception as e:
        # Fallback to default prompt on any error retrieving context
        print(f"Warning: could not retrieve context: {e}")
        prompt = """
You are an expert customer-experience designer tasked with creating comprehensive escalation guidelines for a support chatbot when no existing content is available. Generate a set of clear and concise guidelines for when the chatbot should hand off a conversation to a customer care representative. The guidelines should cover:

Complex Issues Beyond Chatbot's Scope (account/billing/technical issues)
Specific Product or Service Support
Escalation Requests (user demands human assistance)
Missing Content or Knowledge Gaps
Inquiries Needing Deep Explanation

Provide output in structured guidance points with section titles. Do not include ### or numbers anywhere. Use dashes for each item.
"""

    # Call the GPT API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    guidance_text = response.choices[0].message.content

    # Split the guidance text by blank lines into sections
    sections = [sec.strip() for sec in guidance_text.split("\n\n") if sec.strip()]

    # Build MongoDB documents for the first five sections
    guidance_entries = []
    for idx, section in enumerate(sections[:5], start=1):
        guidance_entries.append({
            "chatbot_id": ObjectId(chatbot_id),
            "version_id": ObjectId(version_id),
            "section_title": section.split("\n", 1)[0].strip(),
            "description": "\n".join(section.split("\n")[1:]).strip(),
            "category_name": "handoff",
            "source_type": "ai",
            "is_enabled": False
        })

    # Insert into MongoDB if any entries exist
    if guidance_entries:
        collection.insert_many(guidance_entries)
        print(f"Inserted {len(guidance_entries)} handoff's into MongoDB.")

    return guidance_entries
