import pprint
from bson import ObjectId
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

from Databases.mongo import mongo_crud  # centralized mongo_crud import


# ==== CONFIG ====
DB_NAME = "ChatbotDB-DEV"
VECTOR_DB_PATH = "faiss_index_faq"
COLLECTION_GUIDANCE = "guidanceflows"
COLLECTION_HANDOFF = "handoffscenarios"
COLLECTION_FAQ = "faqs"
COLLECTION_BUZZWORDS = "handoffbuzzwords"


def mongo_operation(operation, collection_name, query=None, projection=None, update=None):
    """
    Helper function to call mongo_crud with optional projection and update support.
    """
    if operation == "read" and projection is not None:
        # Use 'readV2' operation for filter + projection queries
        return mongo_crud(
            host=None,
            port=None,
            db_name=DB_NAME,
            collection_name=collection_name,
            operation="readV2",
            query=[query or {}, projection]
        )
    else:
        return mongo_crud(
            host=None,
            port=None,
            db_name=DB_NAME,
            collection_name=collection_name,
            operation=operation,
            query=query or {},
            update=update or {}
        )


def fetch_data(request_body):
    """
    Fetches filtered data from specified collections based on chatbot_id, version_id, and collections.
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
        guidance_data = mongo_operation(
            operation="readV2",
            collection_name=COLLECTION_GUIDANCE,
            query=query,
            projection={"_id": 0, "section_title": 1, "description": 1}
        )
        result["guidanceflows"] = guidance_data

    if "handoff" in requested_collections:
        handoff_data = mongo_operation(
            operation="readV2",
            collection_name=COLLECTION_HANDOFF,
            query=query,
            projection={"_id": 0, "description": 1}
        )
        result["handoffscenarios"] = handoff_data

    if "handoffbuzzwords" in requested_collections:
        buzzword_data = mongo_operation(
            operation="readV2",
            collection_name=COLLECTION_BUZZWORDS,
            query=query,
            projection={"_id": 0, "buzzwords": 1}
        )
        result["handoffbuzzwords"] = buzzword_data

    return result


def fetch_faqs_and_create_vector(chatbot_id_str, version_id_str):
    """
    Fetches FAQs for given chatbot and version IDs,
    creates a FAISS vector store using LangChain embeddings.
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

    faq_data = mongo_operation(
        operation="read",
        collection_name=COLLECTION_FAQ,
        query=query,
        projection=projection
    )

    if not faq_data:
        print("No FAQ data found.")
        return []

    return create_and_store_vector_db(faq_data)


def create_documents(faqs):
    """Converts FAQ dicts into LangChain Document objects."""
    documents = []
    for faq in faqs:
        question = faq.get("question")
        answer = faq.get("answer")
        if question and answer:
            content = f"Q: {question}\nA: {answer}"
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
    return "Retraining Done"


# ==== Example Usage ====

request_body = {
    "chatbot_id": "6842906726c8b20f873bee6b",
    "version_id": "6842906726c8b20f873bee6f",
    "collection_name": ["guidance", "handoff", "handoffbuzzwords"]
}

merged_result = fetch_data(request_body)
pprint.pprint(merged_result)
