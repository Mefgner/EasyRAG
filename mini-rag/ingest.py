from qdrant_client import QdrantClient
from qdrant_client.http.models import (Distance, PointStruct, VectorParams)
from sentence_transformers import SentenceTransformer
from typing import Optional
from pathlib import Path
import uuid
import tools
import dotenv
import os
import re

dotenv.load_dotenv()

DIM = tools.embeder.get_sentence_embedding_dimension()
NORMALIZE = False

DISTANCE = Distance.COSINE
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "mini-rag"
DOCS_PATH = os.getenv("DOCS_PATH")
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR.joinpath(DOCS_PATH if DOCS_PATH else 'data', 'docs')
qdc = QdrantClient(url=QDRANT_URL)

CHUNK_SIZE = 1600
OVERLAP = int(CHUNK_SIZE / 5)

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
def create_point(text: str, file_name: str, chunk_order: int = -1, raw_additional_meta: Optional[list[str]] = None):
    vector = tools.embed(text)
    
    payload: dict[str, int | str] = {"text": text, "file": file_name}
    
    if chunk_order >= 0:
        payload.update({"order": chunk_order})
        
    if raw_additional_meta:
        additional_meta = {key.lower(): value for key, value in [meta.split(': ') for meta in raw_additional_meta]}
        payload.update(additional_meta)
    
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload=payload
    )
    
    return point


META_RE = re.compile(r"^.{5,}:.{5,}$", re.M)
WHITE_SPACE_RE = re.compile(r"[.]|\n{2,}|\?|!")

points = []

for file_path in DOCS_DIR.glob("*.txt"):
    file = file_path.read_text("utf-8")
    content = file.strip()

    all_metas = META_RE.findall(content, endpos=512)
    all_metas = all_metas[:3]
    content = META_RE.sub("", content, 3)

    if len(content.strip()) > CHUNK_SIZE:
        sentences = WHITE_SPACE_RE.split(content)
        
        chunk: list[str] = []
        i = 0
        while sentences:
            chunk.append(sentences.pop(0))
            
            if len('. '.join(chunk)) >= CHUNK_SIZE:
                points.append(create_point('. '.join(chunk), str(file_path.name), i, all_metas))
                chunk = chunk[-1:]
                if len(chunk[0]) >= OVERLAP:
                    chunk[0] = chunk[0][:OVERLAP]
                i += 1
                
        if len('. '.join(chunk)) > OVERLAP:
            points.append(create_point('. '.join(chunk), str(file_path.name), i, all_metas))

    else:
        points.append(create_point(content, str(file_path.name), -1, all_metas))

qdc.upsert(QDRANT_COLLECTION, points)