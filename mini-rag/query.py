from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tools import embed

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "mini-rag"
qdc = QdrantClient(url=QDRANT_URL)


def query(text: str):
    query_vector = embed(text)
    
    # print("query vector:", query_vector[:5], "...")  # Print first 5 elements for brevity
    
    response = qdc.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=3,
    )

    dict_response = response.model_dump()
    points = dict_response["points"]
    
    if not points:
        return {}, {}
    
    # print("similarity scores: ")
    # print([(point["score"], point["payload"]['file']) for point in points])

    scores = {point["score"]: point["payload"]['file'] for point in points}
    payloads = {point["id"]: point["payload"] for point in points}

    return payloads, scores