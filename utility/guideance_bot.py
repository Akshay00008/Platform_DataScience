import os
from dotenv import load_dotenv
from bson import ObjectId
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from Databases.mongo import mongo_crud  # centralized mongo_crud import

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client and embeddings
client = OpenAI(api_key=openai_api_key)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Constants for DB and collection
#DB_NAME = "ChatbotDB-DEV"
COLLECTION_GUIDANCE = "guidanceflows"

def mongo_operation(operation, collection_name=COLLECTION_GUIDANCE, query=None, update=None):
    """Wrapper for mongo_crud without explicit host/port."""
    return mongo_crud(
      
        collection_name=collection_name,
        operation=operation,
        query=query or {},
        update=update or {}
    )

def load_faiss_index(chatbot_id, version_id):
    """Load the FAISS index fresh from disk each time."""
    import os
    faiss_index_dir = "/home/bramhesh_srivastav/Platform_DataScience/faiss_indexes"
    faiss_index_website = f"{chatbot_id}_{version_id}_faiss_index_website"
    faiss_path = os.path.join(faiss_index_dir, faiss_index_website)
    return FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)

def fetch_vector_content(chatbot_id, version_id, query="overview", k=25):
    """Perform similarity search with fresh FAISS index and return concatenated content."""
    vectorstore = load_faiss_index(chatbot_id, version_id)
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

def generate_guidance(content):
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
"""

    if content and content.strip():
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

Please extract and format the operational behavioral guidelines from the above content.

No content is provided. In this case, generate a comprehensive set of standard operational behavioral guidelines suitable for a professional company assistant bot. Ensure all four required sections are filled with realistic, industry-appropriate rules, without leaving any section empty or giving 
Donot provide responses like this 
"- No specific communication standards were outlined in the provided content."

give some content in each section.No Empty content in any section.responses like this 
"- No specific communication standards were outlined in the provided content."
"""
    else:
        prompt = base_prompt + """

No content is provided. In this case, generate a comprehensive set of standard operational behavioral guidelines suitable for a professional company assistant bot. Ensure all four required sections are filled with realistic, industry-appropriate rules, without leaving any section empty.
"""

    


    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content

def parse_guidance(text, chatbot_id, version_id):
    """Parse structured guidance text into documents suitable for Mongo insertion."""
    sections = text.strip().split("\n\n")
    parsed = []
    try:
        chatbot_oid = ObjectId(chatbot_id)
        version_oid = ObjectId(version_id)
    except Exception as e:
        print("Invalid ObjectId:", e)
        return []

    for section in sections:
        lines = section.strip().split("\n")
        if len(lines) < 2:
            continue
        title = lines[0].strip()
        explanation = "\n".join(lines[1:]).strip()

        parsed.append({
            "chatbot_id": chatbot_oid,
            "version_id": version_oid,
            "section_title": title,
            "category_name": "New",
            "ai_category_name": "Old",
            "source_type": "ai",
            "description": explanation,
            "is_enabled": False
        })
    return parsed

def save_guidance_to_mongo(guidance_docs):
    if not guidance_docs:
        print("No guidance to store.")
        return
    result = mongo_operation(operation="insertmany", query=guidance_docs)
    print(f"Inserted {len(result.inserted_ids)} guidance sections.")

def run_guidance_pipeline(chatbot_id, version_id, query="overview"):
    content = None
    try:
        content = fetch_vector_content(chatbot_id, version_id, query=query)
    except Exception as e:
        print(f"Warning: fetch_vector_content failed: {str(e)}")
        print("Continuing with default guidance generation...")
        content = None

    structured_text = generate_guidance(content)
    structured_docs = parse_guidance(structured_text, chatbot_id, version_id)
    save_guidance_to_mongo(structured_docs)
    return structured_docs
