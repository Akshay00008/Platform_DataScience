from pymongo import MongoClient
from bson import ObjectId
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

def fetch_faqs_from_mongo(uri, db_name, collection_name, chatbot_id_str, version_id_str):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]

    # Convert string IDs to ObjectId
    chatbot_id = ObjectId(chatbot_id_str)
    version_id = ObjectId(version_id_str)

    query = {
        "chatbot_id": chatbot_id,
        "version_id": version_id
    }

    projection = {
        "_id": 0,
        "question": 1,
        "answer": 1
    }

    return list(collection.find(query, projection))

def create_documents(faqs):
    documents = []
    for faq in faqs:
        if faq.get("question") and faq.get("answer"):
            content = f"Q: {faq['question']}\nA: {faq['answer']}"
            documents.append(Document(page_content=content))
    return documents

def main():
    MONGO_URI = "mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017"
    DB_NAME = "ChatbotDB"
    COLLECTION_NAME = "faqs"
    # These must be valid ObjectId strings from your MongoDB collection
    CHATBOT_ID_STR = "6842906726c8b20f873bee6b"
    VERSION_ID_STR = "6842906726c8b20f873bee6f"
    VECTOR_DB_PATH = "faiss_index_faq"

    faqs = fetch_faqs_from_mongo(MONGO_URI, DB_NAME, COLLECTION_NAME, CHATBOT_ID_STR, VERSION_ID_STR)
    if not faqs:
        print("No matching FAQs found.")
        return

    docs = create_documents(faqs)
    splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    vectorstore.save_local(VECTOR_DB_PATH)
    print(f"Vector DB created at '{VECTOR_DB_PATH}'")

if __name__ == "__main__":
    main()
