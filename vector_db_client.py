import chromadb
from chromadb.api.models.Collection import Collection
from config import DB_COLLECTION, DB_HOST, DB_PORT

def get_collection() -> Collection:
    client = chromadb.HttpClient(host=DB_HOST, port=DB_PORT)
    return client.get_collection(DB_COLLECTION)
