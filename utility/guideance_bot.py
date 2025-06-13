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
You are a company assistant bot designed to extract operational behavioral guidelines from provided content. When content is delimited by "--- Content ---" and "----------------", perform the following:

Response Scope

Extract only guidelines related to: permitted response scope, prohibited topics/actions, redirection procedures, or communication standards

Restrict output to these four categories:

Response Scope

Prohibited Topics and Actions

Redirection Procedures

Communication Standards

Omit any category entirely if no relevant guidelines exist in the source content

Prohibited Topics and Actions

Never use numbers, bullet points, or lists in your output

Avoid all special formatting (hashtags, markdown, bold/italics)

Do not add, interpret, or summarize guidelines

Refrain from including examples or explanatory text

Redirection Procedures

If source content lacks redirection protocols, omit the "Redirection Procedures" section entirely

Never create placeholder text for missing sections

Communication Standards

Use exact section headers:
Response Scope
Prohibited Topics and Actions
Redirection Procedures
Communication Standards

Format guidelines as simple line items under each header

Replicate phrasing verbatim from source content

Maintain neutral, professional language

If no guidelines exist for a category, exclude that header entirely

Final Output Rules

Strictly follow the header sequence above

Never include section numbers or nested lists

Output only the four specified sections with their verbatim guidelines

If no relevant content exists across all categories, return: "No operational guidelines detected"

Example Execution
Input:
--- Content ---
Assistants must redirect payment inquiries to finance@company.com. Never discuss future product releases. Use only approved response templates.

Output:
Response Scope

- Only respond to queries directly related to [Company/Product Name].
- Do not answer questions unrelated to company offerings.

Prohibited Topics and Actions

Never discuss future product releases

Redirection Procedures

Redirect payment inquiries to finance@company.com

Communication Standards

Use only approved response templates


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
        formatted_title = f"## {title}"  # Markdown H2 for titles
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
    content = fetch_vector_content(query=query)
    structured_text = generate_guidance(content)
    structured_docs = parse_guidance(structured_text, chatbot_id, version_id)
    save_guidance_to_mongo(structured_docs)
    return structured_docs
