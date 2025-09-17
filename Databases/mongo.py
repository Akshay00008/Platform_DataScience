import pymongo
from bson import ObjectId
from bson.json_util import dumps
import json

def Bot_Retrieval(chatbot_id, version_id):
    client = pymongo.MongoClient("mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")
    db = client["ChatbotDB-DEV"]
    collection = db['chatbotversions']
    
    # If your IDs are ObjectId in MongoDB, convert strings to ObjectId
    try:
        chatbot_obj_id = ObjectId(chatbot_id)
        version_obj_id = ObjectId(version_id)
    except Exception:
        # If IDs are stored as strings, skip conversion
        chatbot_obj_id = chatbot_id
        version_obj_id = version_id
    
    query = {"chatbot_id": chatbot_obj_id, "version_id": version_obj_id}
    documents_cursor = collection.find(query)
    print(documents_cursor)
    print("status")
    documents = list(documents_cursor)  # list of dicts (BSON documents)

    print("**************")
    print(documents)
    
    if not documents:
        return {"error": "No documents found for given chatbot_id and version_id"}
    
    # Use bson.json_util.dumps to serialize ObjectId and other BSON types properly
    json_data = dumps(documents)  # returns a JSON string
    
    # Optionally convert JSON string back to Python dict/list:
    import json
    parsed_json = json.loads(json_data)
    
    return parsed_json


    

def website_tag_saving(website_taggers,chatbot_id,version_id):
    client = pymongo.MongoClient("mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")
    db = client["Chatbot-Backup"]
    collection = db['website_tags']

    print("website_taggers")
    if isinstance(website_taggers, list):
        # If it's a list of tags/documents
        collection.insert_many(website_taggers)
    elif isinstance(website_taggers, dict):
        # If it's a single document
        collection.insert_one(website_taggers)
    else:
        print("Invalid data format for MongoDB insert")
        raise ValueError("Invalid data format for MongoDB insert")
       

    print("Tags inserted successfully.")


def company_Retrieval():
    # Connect to the MongoDB client
    client = pymongo.MongoClient("mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")
    db = client["Chatbot-Backup"]
    collection = db['companies']
    
    # Retrieve all documents from the collection
    documents = list(collection.find())  # This retrieves all documents in the collection
    
    print("**************")
    print(documents)
    
    # If no documents are found, return an error message
    if not documents:
        return {"error": "No documents found in the collection"}
    
    # Use bson.json_util.dumps to serialize ObjectId and other BSON types properly
    json_data = dumps(documents)  # Converts the list of documents to a JSON string
    
    # Optionally, you can convert the JSON string back to a Python dict/list:
    parsed_json = json.loads(json_data)
    
    return parsed_json
