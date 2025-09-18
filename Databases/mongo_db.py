import json
from bson import ObjectId
from bson.json_util import dumps
from Databases.mongo import mongo_crud


DB_NAME_CHATBOT = "ChatbotDB-DEV"
DB_NAME_BACKUP = "Chatbot-Backup"


def Bot_Retrieval(chatbot_id, version_id):
    try:
        chatbot_obj_id = ObjectId(chatbot_id)
        version_obj_id = ObjectId(version_id)
    except Exception:
        chatbot_obj_id = chatbot_id
        version_obj_id = version_id

    query = {"chatbot_id": chatbot_obj_id, "version_id": version_obj_id}

    documents = mongo_crud(
        host=None,
        port=None,
        db_name=DB_NAME_CHATBOT,
        collection_name='chatbotversions',
        operation='read',
        query=query
    )

    if not documents:
        return {"error": "No documents found for given chatbot_id and version_id"}

    # Serialize to JSON string with proper BSON handling
    json_data = dumps(documents)

    # Deserialize back to Python object
    parsed_json = json.loads(json_data)

    return parsed_json


def website_tag_saving(website_taggers, chatbot_id=None, version_id=None):
    """
    Save website tags documents into MongoDB collection 'website_tags' in Chatbot-Backup DB.
    chatbot_id and version_id parameters are optional here.
    """
    if not isinstance(website_taggers, (list, dict)):
        raise ValueError("Invalid data format for MongoDB insert: must be list or dict")

    operation = 'create'  # for insert_one or insert_many

    if isinstance(website_taggers, list):
        # Insert many
        mongo_crud(
            host=None,
            port=None,
            db_name=DB_NAME_BACKUP,
            collection_name='website_tags',
            operation='insertmany',
            query=website_taggers
        )
    else:
        # Insert one
        mongo_crud(
            host=None,
            port=None,
            db_name=DB_NAME_BACKUP,
            collection_name='website_tags',
            operation='create',
            query=website_taggers
        )

    print("Tags inserted successfully.")


def company_Retrieval():
    documents = mongo_crud(
        host=None,
        port=None,
        db_name=DB_NAME_BACKUP,
        collection_name='companies',
        operation='read'
    )

    if not documents:
        return {"error": "No documents found in the collection"}

    json_data = dumps(documents)
    parsed_json = json.loads(json_data)

    return parsed_json
