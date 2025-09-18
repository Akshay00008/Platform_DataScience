import os
import json
import re
import logging
from dotenv import load_dotenv
from bson import ObjectId

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from Databases.mongo import mongo_crud

load_dotenv()

# Constants
DB_NAME = "ChatbotDB-DEV"
COLLECTION_FAQS = "faqs"
COLLECTION_CATALOGUE = "catelogue"

# Initialize OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

def mongo_operation(operation, collection_name=COLLECTION_FAQS, query=None, update=None, start=0, stop=10):
    """Helper to call mongo_crud with fixed db and no host/port needed."""
    return mongo_crud(
        host=None,
        port=None,
        db_name=DB_NAME,
        collection_name=collection_name,
        operation=operation,
        query=query or {},
        update=update or {},
        start=start,
        stop=stop
    )

def load_faiss_index(chatbot_id, version_id, target_vector):
    faiss_index_dir = "/home/bramhesh_srivastav/platformdevelopment/faiss_indexes"
    faiss_index_filename = f"{chatbot_id}_{version_id}_faiss_index"
    faiss_index_website = f"{chatbot_id}_{version_id}_faiss_index_website"
    faiss_path_1 = os.path.join(faiss_index_dir, faiss_index_filename)    
    faiss_path_2 = os.path.join(faiss_index_dir, faiss_index_website)

    try:
        if 'faq' in target_vector:
            faiss_path = faiss_path_1
            print(f"Loading FAISS index from: {faiss_path}")
        elif 'website' in target_vector:
            faiss_path = faiss_path_2
            print(f"Loading FAISS index from: {faiss_path}")
        else:
            raise ValueError("Invalid vector type. Please use 'faq' or 'website'.")
        return FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        return f"Error loading FAISS index: {e}"

def search_faiss(query, faiss_load, k=10):
    results = faiss_load.similarity_search(query, k=k)
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
        model="gpt-4o-mini",
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
        model="gpt-4o-mini",
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
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        category = response.choices[0].message.content.strip()
        faq["ai_category_name"] = category or "product"
    return faq_list

def save_faqs_to_mongo(faq_list, chatbot_id, version_id):
    if not faq_list:
        print("No FAQs to save.")
        return 0
    try:
        chatbot_oid = ObjectId(chatbot_id)
        version_oid = ObjectId(version_id)
    except Exception as e:
        print(f"Invalid ObjectId: {e}")
        return 0

    for index, faq in enumerate(faq_list, start=1):
        faq["chatbot_id"] = chatbot_oid
        faq["version_id"] = version_oid
        faq["is_enabled"] = False
        faq["category_name"] = "New"
        faq["ai_category_name"] = "Product"
        faq["source_type"] = "ai"

# Exclude the 0th index from faq_list
    print(faq_list)
    faq_list_to_insert = faq_list[1:]

# Insert into MongoDB
    result = mongo_operation(operation="insertmany", query=faq_list)
    print(f"Inserted {len(result.inserted_ids)} FAQs into MongoDB.")
    return len(result.inserted_ids)

def generate_tags_and_buckets_from_json(chunks, chatbot_id, version_id, url, target_count=50):
    joined_chunks = "\n\n".join(chunks[:30])
    chatbot_oid = ObjectId(chatbot_id)
    version_oid = ObjectId(version_id)
    collection_name = COLLECTION_CATALOGUE

    prompt = f"""
I have the following content extracted from a webpage:

{joined_chunks}

Please generate relevant tags based on this content, categorizing them into appropriate buckets. The tags should describe key topics, products, services, or concepts mentioned on the page, and each tag should be categorized into a relevant bucket. Example buckets could be 'Products', 'Applications', 'Services', 'Industries', 'Solutions', 'Others', etc.

The output should be in the following JSON format:
{{
  "Catalogue Name 1": {{
    "Name 1": "Description of the concept, product, service, or industry.",
    "Name 2": "Description of the concept, product, service, or industry."
  }},
  "Catalogue Name 2": {{
    "Name 1": "Description of the concept, product, service, or industry.",
    "Name 2": "Description of the concept, product, service, or industry."
  }}
}}

Here is an example format of the JSON output:
{{
  "Industries": {{
    "Semiconductor": "The semiconductor industry involves the design and fabrication of microchips used in various devices.",
    "Surface Finishing": "Surface finishing refers to processes that improve the appearance, durability, and wear resistance of materials."
  }},
  "Products": {{
    "XYZ Product": "A high-performance product designed to meet the needs of modern manufacturing."
  }},
  "Solutions": {{
    "Cloud-based Solution": "A scalable solution that enables businesses to migrate their operations to the cloud."
  }}
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        category = response.choices[0].message.content.strip()

        print("Raw output:\n", category)

        # Clean the string: Remove markdown and extra characters like backticks
        cleaned_category = re.sub(r'```json|```', '', category).strip()

        # Try parsing the cleaned JSON string
        category_obj = json.loads(cleaned_category)

        # print("Parsed JSON object:\n", category_obj)

    except json.JSONDecodeError as e:
        logging.error(f"JSON Decode Error: {e}")
        category_obj = {}
    except Exception as e:
        logging.error(f"Failed to generate or parse tags and buckets: {e}")
        category_obj = {}


        # print("Parsed JSON object:\n", category_obj)

    except json.JSONDecodeError as e:
        logging.error(f"JSON Decode Error: {e}")
        category_obj = {}
    except Exception as e:
        logging.error(f"Failed to generate or parse tags and buckets: {e}")
        category_obj = {}



    print("tags_and_buckets:", category_obj)

    document = {
        "chatbot_id": chatbot_oid,
        "version_id": version_oid,
        "Catalogue": category_obj,
        "url": url
    }

    try:
        result = mongo_crud(
            host=None,
            port=None,
            db_name=DB_NAME,
            collection_name=collection_name,
            operation="create",
            query=document
        )
        return {"tags_and_buckets": category_obj}
    except Exception as e:
        logging.error(f"An error occurred during saving tags and buckets: {e}")
        return {"tags_and_buckets": {}, "error": str(e)}

def translate_welcome_message(message: str, lang: str) -> str:
    prompt = f"""You are an agent that converts the given welcome message: "{message}" into the required language: {lang}. Make it sound natural and welcoming."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"
