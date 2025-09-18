import pymongo
from urllib.parse import quote as urlquote


def mongo_crud(host, port, db_name, collection_name, operation, query={}, update={}, start=0, stop=10):
    client = pymongo.MongoClient(r"mongodb://dev:N47309HxFWE2Ehc@35.209.224.122:27017")
    db = client[db_name]
    collection = db[collection_name]
    
    if operation == "create":
        return collection.insert_one(query).inserted_id
    elif operation == "insertmany":
        return collection.insert_many(query)
    elif operation == "read":
        return list(collection.find(query))
    elif operation == "readV2":
        return list(collection.find(filter=query[0],projection=query[1]))
    elif operation == "findone":
        return collection.find_one(query)
    elif operation == "update":
        return collection.find_one_and_update(query, {"$set": update})
    elif operation == "updatemany":
        return collection.update_many(query, update=update)
    elif operation == "delete":
        return collection.delete_one(query).deleted_count
    elif operation == "deletemany":
        return collection.delete_many({})
    elif operation == "drop":
        return collection.drop()
    elif operation == "bulkWrite":
        return collection.bulk_write(update,ordered=False,bypass_document_validation=True)
    elif operation == "paginatedQuery":
        return list(collection.find(query).skip(start).limit(stop - start))
    else:
        return "Invalid Operation"


# 4$ggw8@3i!t5H1z_3R/=bj]f['&f/0x!