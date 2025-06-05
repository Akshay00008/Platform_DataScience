import pymongo
from bson import ObjectId
from bson.json_util import dumps

def Bot_Retrieval(chatbot_id, version_id):
    client = pymongo.MongoClient("mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")
    db = client["ChatbotDB"]
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
    
    documents = list(documents_cursor)  # list of dicts (BSON documents)
    
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
    db = client["ChatbotDB"]
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

