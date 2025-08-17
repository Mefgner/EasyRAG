from sentence_transformers import SentenceTransformer

embeder = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str):
    return embeder.encode(text).tolist()