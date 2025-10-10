import os
import pymongo
from dotenv import load_dotenv

load_dotenv()


# Read full connection string from backend-secrets
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "defaultdb")


def mongo_crud(collection_name, operation, query=None, update=None, start=0, stop=10, projection=None, **kwargs):
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI environment variable is not set")

    client = pymongo.MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[collection_name]

    query = query or {}
    update = update or {}

    if operation == "create":
        return collection.insert_one(query).inserted_id
    elif operation == "insertmany":
        return collection.insert_many(query)
    elif operation == "read":
        return list(collection.find(query))
    elif operation == "readV2":
        if isinstance(query, list) and len(query) == 2:
            filter_q, proj_q = query
            return list(collection.find(filter=filter_q, projection=proj_q))
        else:
            return list(collection.find(filter=query or {}, projection=projection))
    elif operation == "findone":
        return collection.find_one(query)
    elif operation == "update":
        # ✅ Check if update already contains MongoDB operators
        if any(key.startswith("$") for key in update.keys()):
            update_doc = update
        else:
            update_doc = {"$set": update}

        # Use update_one if you need upsert or return info
        return collection.update_one(query, update_doc, **kwargs)
    elif operation == "updatemany":
        if any(key.startswith("$") for key in update.keys()):
            update_doc = update
        else:
            update_doc = {"$set": update}
        return collection.update_many(query, update_doc, **kwargs)
    elif operation == "delete":
        return collection.delete_one(query).deleted_count
    elif operation == "deletemany":
        return collection.delete_many({})
    elif operation == "drop":
        return collection.drop()
    elif operation == "bulkWrite":
        return collection.bulk_write(update, ordered=False, bypass_document_validation=True)
    elif operation == "paginatedQuery":
        return list(collection.find(query).skip(start).limit(stop - start))
    else:
        return "Invalid Operation"
