import pymongo
from bson import ObjectId
import pprint
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

# ==== CONFIG ====
MONGO_URI = "mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017"
DB_NAME = "ChatbotDB"
VECTOR_DB_PATH = "faiss_index_faq"

# ==== DB CONNECTION ====
mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
guidance_collection = db["guidanceflows"]
handoff_collection = db["handoffscenarios"]
faq_collection = db["faqs"]

# ==== CORE FUNCTION ====

def fetch_data(request_body):
    """
    Fetches filtered data from guidanceflows and handoffscenarios collections 
    based on chatbot_id, version_id, and selected collections.
    """
    chatbot_id_str = request_body.get("chatbot_id")
    version_id_str = request_body.get("version_id")
    requested_collections = request_body.get("collection_name", [])

    if not chatbot_id_str or not version_id_str or not requested_collections:
        raise ValueError("Missing required fields: chatbot_id, version_id, or collection_name")

    try:
        chatbot_id = ObjectId(chatbot_id_str)
        version_id = ObjectId(version_id_str)
    except Exception as e:
        raise ValueError(f"Invalid ObjectId format: {e}")

    query = {
        "chatbot_id": chatbot_id,
        "version_id": version_id,
        "is_enabled": True
    }

    result = {}

    if "guidance" in requested_collections:
        guidance_data = list(guidance_collection.find(query, {
            "_id": 0,
            "section_title": 1,
            "content": 1
        }))
        result["guidanceflows"] = guidance_data

    if "handoff" in requested_collections:
        handoff_data = list(handoff_collection.find(query, {
            "_id": 0,
            "guidance": 1
        }))
        result["handoffscenarios"] = handoff_data

    return result


def fetch_faqs_and_create_vector(chatbot_id_str, version_id_str):
    """
    Fetches FAQs for given chatbot and version IDs,
    creates a FAISS vector store using LangChain.
    """
    try:
        chatbot_id = ObjectId(chatbot_id_str)
        version_id = ObjectId(version_id_str)
    except Exception as e:
        raise ValueError(f"Invalid ObjectId format: {e}")

    query = {
        "chatbot_id": chatbot_id,
        "version_id": version_id
    }

    projection = {
        "_id": 0,
        "question": 1,
        "answer": 1
    }

    faq_data = list(faq_collection.find(query, projection))

    if not faq_data:
        print("No FAQ data found.")
        return []

    # Create FAISS vector store
    result = create_and_store_vector_db(faq_data)

    return result


# ==== EMBEDDING & VECTOR DB LOGIC ====

def create_documents(faqs):
    """Converts FAQ dicts into LangChain Document objects."""
    documents = []
    for faq in faqs:
        if faq.get("question") and faq.get("answer"):
            content = f"Q: {faq['question']}\nA: {faq['answer']}"
            documents.append(Document(page_content=content))
    return documents

def create_and_store_vector_db(faqs):
    """Creates a FAISS vector DB from FAQ data."""
    docs = create_documents(faqs)
    splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    return ("Retraining Done")

# ==== USAGE ====


    request_body = {
        "chatbot_id": "6842906726c8b20f873bee6b",
        "version_id": "6842906726c8b20f873bee6f",
        "collection_name": ["guidance", "handoff", "faqs"]
    }

    merged_result = fetch_data(request_body)
    pprint.pprint(merged_result)
