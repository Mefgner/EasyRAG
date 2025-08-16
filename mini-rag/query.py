from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

embeder = SentenceTransformer("all-MiniLM-L6-v2")
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "mini-rag"
qdc = QdrantClient(url=QDRANT_URL)

def embed(text: str):
    return embeder.encode(text).tolist()

def query(text: str):
    query_vector = embed(text)
    
    # print("query vector:", query_vector[:5], "...")  # Print first 5 elements for brevity
    
    response = qdc.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=5,
        score_threshold=0.5,
    )

    dict_response = response.model_dump()
    points = dict_response["points"]
    
    # if points:
    #     print("similarity scores: ")
    #     print([(point["score"], point["payload"]['file']) for point in points])

    payloads = {point["id"]: point["payload"] for point in points}
    
    return payloads