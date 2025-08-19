from qdrant_client import QdrantClient
from qdrant_client.http.models import (Distance, PointStruct, VectorParams)
from sentence_transformers import SentenceTransformer
from pathlib import Path
import uuid
import tools

DIM = tools.embeder.get_sentence_embedding_dimension()
NORMALIZE = False

DISTANCE = Distance.COSINE
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "mini-rag"
DOCS_DIR = Path("./data/docs")
qdc = QdrantClient(url=QDRANT_URL)

TEST_MODE = True

if TEST_MODE:
    try:
        qdc.delete_collection(QDRANT_COLLECTION)
    finally:
        pass

if not qdc.collection_exists(QDRANT_COLLECTION):
    qdc.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=DIM if DIM else 384,  # Default size for MiniLM
            distance=DISTANCE,
        )
    )


# TODO:
    # - make better chunking with overlapping
    # - add chunk id or chunk order when working with big documents/contents
    # - add metadata like lang
def add_embedding(text: str, file_name: str, chunk_order: int = -1):
    vector = tools.embed(text)
    
    payload: dict[str, int | str] = {"text": text, "file": file_name}
    
    if chunk_order >= 0:
        payload.update({"order": chunk_order})
    
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload=payload
    )
    
    qdc.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[point]
    )


for file_path in DOCS_DIR.glob("*.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read().strip()
        
        if len(content.strip()) > 1000:
            sentences = content.split(". ")
            
            chunk = ""
            i = 0
            while sentences:
                chunk += sentences.pop(0) + ". "
                
                if len(chunk) >= 750:
                    add_embedding(chunk, str(file_path.name), i)
                    chunk = ""
                    i += 1
                    
            if chunk:
                add_embedding(chunk, str(file_path.name), i)

        else:
            add_embedding(content, str(file_path.name))
