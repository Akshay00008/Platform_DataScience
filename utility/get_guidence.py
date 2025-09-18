from bson import ObjectId
import pprint
from Databases.mongo import mongo_crud

# Constants for DB and collections
DB_NAME = "ChatbotDB-DEV"
COLLECTION_GUIDANCE = "guidanceflows"
COLLECTION_HANDOFF = "handoffscenarios"

def mongo_operation(collection_name, operation, query=None, projection=None):
    """Helper to call mongo_crud without host and port, with optional projection."""
    if operation == "read":
        # mongo_crud 'readV2' supports projection (filter, projection)
        if projection:
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
                query=query or {}
            )
    else:
        return mongo_crud(
            host=None,
            port=None,
            db_name=DB_NAME,
            collection_name=collection_name,
            operation=operation,
            query=query or {}
        )

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
guidance_data = mongo_operation(
    collection_name=COLLECTION_GUIDANCE,
    operation="readV2",
    query=query,
    projection={"_id": 0, "section_title": 1, "content": 1}
)

# Get only guidance field from handoffscenarios
handoff_data = mongo_operation(
    collection_name=COLLECTION_HANDOFF,
    operation="readV2",
    query=query,
    projection={"_id": 0, "guidance": 1}
)

# Merge and display
merged_result = {
    "guidanceflows": guidance_data,
    "handoffscenarios": handoff_data
}

pprint.pprint(merged_result)
