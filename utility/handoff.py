import os
from dotenv import load_dotenv
from bson import ObjectId
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

# Import centralized mongo_crud
from Databases.mongo import mongo_crud

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client and embeddings
client = OpenAI(api_key=openai_api_key)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Constants for DB and collection
#DB_NAME = "ChatbotDB-DEV"
COLLECTION_HANDOFF = "handoffscenarios"

def mongo_operation(operation, collection_name=COLLECTION_HANDOFF, query=None, update=None):
    """Unified helper to call mongo_crud without specifying host/port."""
    return mongo_crud(
       
        collection_name=collection_name,
        operation=operation,
        query=query or {},
        update=update or {}
    )

def load_faiss_index(chatbot_id, version_id):
    """Load the FAISS index fresh from disk each call."""
    faiss_index_dir = "/home/bramhesh_srivastav/platformdevelopment/faiss_indexes"
    faiss_index_website = f"{chatbot_id}_{version_id}_faiss_index_website"
    faiss_path = os.path.join(faiss_index_dir, faiss_index_website)
    return FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)

def search_vector_context(chatbot_id, version_id, query, k=30):
    """Perform similarity search with fresh FAISS index and aggregate results."""
    vectorstore = load_faiss_index(chatbot_id, version_id)
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

def generate_handoff_guidance(query, chatbot_id, version_id):
    """Generate structured handoff guidance based on query and vectorstore context."""
    try:
        context = search_vector_context(chatbot_id, version_id, query)
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
        print(f"Warning: context retrieval failed: {e}")
        prompt = """
You are an expert customer-experience designer tasked with creating comprehensive escalation guidelines for a support chatbot when no existing content is available. Generate a set of clear and concise guidelines for when the chatbot should hand off a conversation to a customer care representative. The guidelines should cover:

Complex Issues Beyond Chatbot's Scope (account/billing/technical issues)
Specific Product or Service Support
Escalation Requests (user demands human assistance)
Missing Content or Knowledge Gaps
Inquiries Needing Deep Explanation

Provide output in structured guidance points with section titles. Do not include ### or numbers anywhere. Use dashes for each item.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    guidance_text = response.choices[0].message.content

    # Split guidance by blank line sections
    sections = [sec.strip() for sec in guidance_text.split("\n\n") if sec.strip()]

    # Prepare documents for Mongo insertion
    guidance_entries = []
    for section in sections[:5]:
        lines = section.split("\n")
        if not lines:
            continue
        guidance_entries.append({
            "chatbot_id": ObjectId(chatbot_id),
            "version_id": ObjectId(version_id),
            "section_title": lines[0].strip(),
            "description": "\n".join(lines[1:]).strip(),
            "category_name": "New",
            "source_type": "ai",
            "is_enabled": False
        })

    if guidance_entries:
        mongo_operation(operation="insertmany", collection_name=COLLECTION_HANDOFF, query=guidance_entries)
        print(f"Inserted {len(guidance_entries)} handoff guidance entries into MongoDB.")

    return guidance_entries
