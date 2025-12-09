import pymongo
import os
from dotenv import load_dotenv
load_dotenv()

client = pymongo.MongoClient(os.getenv("MONGO_DB_URL"))
db = client["RitikDB"]
col = db["NetworkData"]

print("Document count:", col.count_documents({}))
