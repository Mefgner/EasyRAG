from qdrant_client import QdrantClient
from qdrant_client.http.models import (Distance, PointStruct, VectorParams)
from sentence_transformers import SentenceTransformer
from pathlib import Path
import uuid

embeder = SentenceTransformer("all-MiniLM-L6-v2")
DIM = embeder.get_sentence_embedding_dimension()
NORMALIZE = False

DISTANCE = Distance.COSINE
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "mini-rag"
DOCS_DIR = Path("./data/docs")
qdc = QdrantClient(url=QDRANT_URL)

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
    # - add chunk id or chunk order when working with big documents/paragraphs
    # - add metadata like lang
def embed(text: str):
    return embeder.encode(text, normalize_embeddings=NORMALIZE).tolist()


def add_embedding(text: str, file_name: str):
    vector = embed(text)
    
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={"text": text, "file": file_name}
    )
    
    qdc.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[point]
    )


for file_path in DOCS_DIR.glob("*.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        for paragraph in file.read().split("\n\n"):
            if len(paragraph.strip()) < 10:
                continue
            
            elif len(paragraph.strip()) > 1000:
                for chunk in [paragraph[i:i + 1000] for i in range(0, len(paragraph), 1000)]:
                    if chunk.strip():
                        add_embedding(chunk, str(file_path.name))

            else:
                add_embedding(paragraph, str(file_path.name))
