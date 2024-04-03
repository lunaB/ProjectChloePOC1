import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os
import time

# load dotenv
load_dotenv()
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
print("[INFO] Load enviroment")


ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=HUGGING_FACE_API_KEY,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = chromadb.PersistentClient(path="chroma")
collection_history = db.get_or_create_collection(name="history", embedding_function=ef) # all history (chat, etc)
collection_inference = db.get_or_create_collection(name="inference", embedding_function=ef) # inference about user
print("[Info] Database loaded")

user_text = "鬼滅の刃"
user_timestemp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

result = collection_history.query(
	query_embeddings=ef(user_text),
	n_results=3
)

print(result["documents"][0])	
print(result["metadatas"][0])

print('\n'.join([f"[{meta['timestemp']}] {meta['type']} : {text}" for text, meta in zip(result["documents"][0], result["metadatas"][0])]))

	
