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


# MongoDB connection
mongo_client = pymongo.MongoClient("mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")
db = mongo_client["ChatbotDB"]
guidance_collection = db["guidanceflows"]

# Function to load FAISS index fresh every time
def load_faiss_index(chatbot_id,version_id,):
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

# Fetch content from vector store
def fetch_vector_content(chatbot_id,version_id,query="overview", k=25):
    """
    Fetch the vector content by performing a similarity search with a fresh FAISS index.
    """
    vectorstore = load_faiss_index(chatbot_id,version_id,)  # Reload FAISS index each time
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

# Generate structured guidance using GPT-4o
def generate_guidance(content):
    prompt = f"""
You are a company assistant bot designed to generate operational behavioral guidelines from provided content. Your task is to extract and clearly format all relevant behavioral restrictions, action instructions, scope limitations, redirection procedures, and communication standards as a numbered list.

Formatting Rules:
- Organize the output into clear section titles, using the following categories (add or adjust as needed):
  - Response Scope
  - Prohibited Topics and Actions
  - Redirection Procedures
  - Communication Standards

Extraction Criteria:
- Extract and format only the guidelines that specify:
  - Permitted response scope
  - Prohibited topics/actions
  - Required redirection procedures
  - Communication standards

Example Output Structure:
Response Scope
   - Only respond to queries directly related to [Company/Product Name].
   - Do not answer questions unrelated to company offerings.

Prohibited Topics and Actions
   - Never discuss pricing or payments.
   - Do not provide legal advice.

Redirection Procedures
   - Redirect billing questions to customer care.
   - Forward legal inquiries to the company’s legal department.

 Communication Standards
   - Maintain professional and respectful language.
   - Reference only official company documentation in responses.

Your task:
Whenever content is provided between "--- Content ---" {content} and "----------------", extract and format the operational behavioral guidelines as it is in the Example Output Structure above donot include numbering in your response numbering is strictly prohobitted also donot use hastags like this simple text ## Response Scope.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content

# Parse structured guidance into a list of documents with proper Markdown
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

        # Proper Markdown formatting
        formatted_title = f"{title}"  # Markdown H2 for titles
        formatted_explanation = f"{explanation}"  # Regular Markdown content

        parsed.append({
            "chatbot_id": chatbot_oid,
            "version_id": version_oid,
            "section_title": formatted_title,
            "category_name": "New",
            "ai_category_name": "Old",
            "source_type": "ai",
            "description": formatted_explanation,
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
    content = fetch_vector_content(chatbot_id,version_id,  query=query)
    structured_text = generate_guidance(content)
    structured_docs = parse_guidance(structured_text, chatbot_id, version_id)
    save_guidance_to_mongo(structured_docs)
    return structured_docs
