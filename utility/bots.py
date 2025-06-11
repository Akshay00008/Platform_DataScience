import os
from dotenv import load_dotenv
import pymongo
import re
from bson import ObjectId

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Load env variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# MongoDB setup
mongo_client = pymongo.MongoClient("mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")
db = mongo_client["ChatbotDB"]
collection = db['faqs']

# FAISS vectorstore path
faiss_path = r"/home/bramhesh_srivastav/Platform_DataScience/website_faiss_index"


def load_faiss_index():
    """
    This function loads the FAISS index fresh every time it is called.
    """
    return FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)


def search_faiss(query, k=10):
    """
    Perform similarity search on the FAISS index and return results.
    This ensures fresh loading of the FAISS index on each search request.
    """
    vectorstore = load_faiss_index()  # Reload FAISS index on each request
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


def categorize_faqs(faq_list, context_chunks):
    """
    For each FAQ, use the website content context to assign a category.
    Returns list of FAQs with added 'category' field.
    """
    context_text = "\n\n".join(context_chunks[:30])
    for faq in faq_list:
        prompt = f"""
You are an AI assistant analyzing FAQs based on the following website content:

{context_text}

Assign a concise and relevant category for this FAQ:

Q: {faq['question']}
A: {faq['answer']}

Return only the category name in one or two words.
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        category = response.choices[0].message.content.strip()
        faq["ai_category_name"] = "product"

    return faq_list


def save_faqs_to_mongo(faq_list, chatbot_id, version_id):
    if not faq_list:
        print("No FAQs to save.")
        return 0
    
    try:
        chatbot_oid = ObjectId(chatbot_id)
        version_oid = ObjectId(version_id)
    except Exception as e:
        print("Invalid ObjectId:", e)
        return 0

    for faq in faq_list:
        faq["chatbot_id"] = chatbot_oid
        faq["version_id"] = version_oid
        faq["is_enabled"] = False
        faq["category_name"] = "New"
        faq["ai_category_name"] = "Product"
        faq["source_type"] = "ai"

    result = collection.insert_many(faq_list)
    print(f"Inserted {len(result.inserted_ids)} FAQs into MongoDB.")
    return len(result.inserted_ids)