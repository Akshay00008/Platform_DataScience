# handoff_bot.py

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
# faiss_path = r"C:\Users\hp\Desktop\Platform_16-05-2025\Platform_DataScience\website_faiss_index"
faiss_path = r"/home/bramhesh_srivastav/Platform_DataScience/website_faiss_index"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)

# OpenAI client
client = OpenAI(api_key=openai_api_key)

def search_vector_context(query, k=30):
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

    try:
        chatbot_oid = ObjectId(chatbot_id)
        version_oid = ObjectId(version_id)
    except Exception as e:
        print("Invalid ObjectId:", e)
        return 0

    guidance_doc = {
        "chatbot_id": chatbot_oid,
        "version_id": version_oid,
        "description": guidance_text,
        "category_name" : "New",
        "source_type" : "ai",
        "is_enabled": False
    }

    collection.insert_one(guidance_doc)
    return guidance_text
