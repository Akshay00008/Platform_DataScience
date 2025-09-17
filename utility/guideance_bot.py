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
db = mongo_client["ChatbotDB-DEV"]
guidance_collection = db["guidanceflows"]

# Function to load FAISS index fresh every time
def load_faiss_index(chatbot_id,version_id,):
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
You are a specialized company assistant bot designed to extract and format operational behavioral guidelines from provided content. Your primary function is to identify, extract, and clearly organize all behavioral restrictions, action instructions, scope limitations, redirection procedures, and communication standards into a structured format.

Core Task:
Extract only the content that specifies operational guidelines and format them into clearly defined sections. Focus exclusively on actionable directives, restrictions, and procedural instructions.

Required Output Structure:
Organize extracted guidelines into these four primary sections (adapt section titles as needed based on content):

Response Scope

[Extract permitted response boundaries and operational limits]

Prohibited Topics and Actions

[Extract forbidden topics, restricted actions, and prohibited behaviors]

Redirection Procedures

[Extract specific instructions for routing queries to appropriate channels]

Communication Standards

[Extract tone requirements, language guidelines, and interaction protocols]

Formatting Requirements:

Use simple text formatting without numbering, bullet points, or hashtag headers

Present each section with plain text section titles

List guidelines using dashes (-) for each item

Maintain the exact wording from the source content

Do not add, interpret, or modify the original guidelines

Processing Instructions:
When content is provided between "--- Content ---" {content} and "----------------":

Scan the content for behavioral guidelines, restrictions, and operational instructions

Categorize findings into the four specified sections

Extract guidelines verbatim without interpretation or modification

Format according to the structure requirements above

Omit any content that doesn't constitute operational behavioral guidelines

Example Output Format:

Response Scope

Only respond to queries directly related to [Company/Product Name]

Do not answer questions unrelated to company offerings

Prohibited Topics and Actions

Never discuss pricing or payments

Do not provide legal advice

Redirection Procedures

Redirect billing questions to customer care

Forward legal inquiries to the company's legal department

Communication Standards

Maintain professional and respectful language

Reference only official company documentation in responses

Important: Extract and present guidelines exactly as written in the source material. Do not summarize, paraphrase, or add interpretative content."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content

def generate_guidance(content):
    # Base prompt without content processing instructions
    base_prompt = """
You are a specialized company assistant bot designed to extract and format operational behavioral guidelines from provided content. Your primary function is to identify, extract, and clearly organize all behavioral restrictions, action instructions, scope limitations, redirection procedures, and communication standards into a structured format.

Core Task:
Extract only the content that specifies operational guidelines and format them into clearly defined sections. Focus exclusively on actionable directives, restrictions, and procedural instructions.

Required Output Structure:
Organize extracted guidelines into these four primary sections (adapt section titles as needed based on content):

Response Scope
[Extract permitted response boundaries and operational limits]

Prohibited Topics and Actions
[Extract forbidden topics, restricted actions, and prohibited behaviors]

Redirection Procedures
[Extract specific instructions for routing queries to appropriate channels]

Communication Standards
[Extract tone requirements, language guidelines, and interaction protocols]

Formatting Requirements:
- Use simple text formatting without numbering, bullet points, or hashtag headers
- Present each section with plain text section titles
- List guidelines using dashes (-) for each item
- Maintain the exact wording from the source content
- Do not add, interpret, or modify the original guidelines

Example Output Format:

Response Scope
- Only respond to queries directly related to [Company/Product Name]
- Do not answer questions unrelated to company offerings

Prohibited Topics and Actions
- Never discuss pricing or payments
- Do not provide legal advice

Redirection Procedures
- Redirect billing questions to customer care
- Forward legal inquiries to the company's legal department

Communication Standards
- Maintain professional and respectful language
- Reference only official company documentation in responses

Important: Extract and present guidelines exactly as written in the source material. Do not summarize, paraphrase, or add interpretative content.
"""

    # Check if content is available and not empty
    if content and content.strip():
        # Content is available - add content processing instructions
        prompt = base_prompt + f"""

Processing Instructions:
Content is provided between "--- Content ---" and "----------------":
1. Scan the content for behavioral guidelines, restrictions, and operational instructions
2. Categorize findings into the four specified sections
3. Extract guidelines verbatim without interpretation or modification
4. Format according to the structure requirements above
5. Omit any content that doesn't constitute operational behavioral guidelines

--- Content ---
{content}
----------------

Please extract and format the operational behavioral guidelines from the above content."""

    else:
        # No content available - use the prompt itself as guidance
        prompt = base_prompt + """

Since no specific content is provided, generate a comprehensive set of standard operational behavioral guidelines that would be appropriate for a professional company assistant bot. Create realistic guidelines covering all four required sections while maintaining the specified formatting requirements."""

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
# def run_guidance_pipeline(chatbot_id, version_id, query="overview"):
#     content = fetch_vector_content(chatbot_id,version_id,  query=query)
#     structured_text = generate_guidance(content)
#     structured_docs = parse_guidance(structured_text, chatbot_id, version_id)
#     save_guidance_to_mongo(structured_docs)
#     return structured_docs
def run_guidance_pipeline(chatbot_id, version_id, query="overview"):
    content = None
    
    try:
        # Attempt to fetch vector content
        content = fetch_vector_content(chatbot_id, version_id, query=query)
    except Exception as e:
        # Log the error and continue with graceful degradation
        print(f"Warning: fetch_vector_content failed: {str(e)}")
        print("Continuing with default guidance generation...")
        content = None  # Explicitly set to None for clarity
    
    # Generate guidance regardless of content availability
    structured_text = generate_guidance(content)
    structured_docs = parse_guidance(structured_text, chatbot_id, version_id)
    save_guidance_to_mongo(structured_docs)
    
    return structured_docs
