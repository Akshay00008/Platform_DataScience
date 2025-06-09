import pymongo
from bson import ObjectId
import pprint

# MongoDB connection
mongo_client = pymongo.MongoClient("mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")
db = mongo_client["ChatbotDB"]

# Collections
guidance_collection = db["guidanceflows"]
handoff_collection = db["handoffscenarios"]

# Replace with actual ObjectId strings
chatbot_id_str = "6643f31b2eacee1e187b2f0b"
version_id_str = "665cb9ae24412295a9a38287"

# Convert to ObjectId
chatbot_id = ObjectId(chatbot_id_str)
version_id = ObjectId(version_id_str)

# Common query including is_enabled
query = {
    "chatbot_id": chatbot_id,
    "version_id": version_id,
    "is_enabled": True
}

# Get only section_title and content from guidanceflows
guidance_data = list(guidance_collection.find(query, {
    "_id": 0,
    "section_title": 1,
    "content": 1
}))

# Get only guidance field from handoffscenarios
handoff_data = list(handoff_collection.find(query, {
    "_id": 0,
    "guidance": 1
}))

# Merge and display
merged_result = {
    "guidanceflows": guidance_data,
    "handoffscenarios": handoff_data
}

pprint.pprint(merged_result)
