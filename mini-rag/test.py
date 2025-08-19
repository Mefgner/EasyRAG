import query

test_cases = [
    {"input": "What is RAG?", "suggested_files": ["01_rag_basics.txt",]},
    {"input": "What is embeddings?", "suggested_files": ["02_embeddings.txt",]},
    {"input": "Tell about vector scores", "suggested_files": ["03_vector_stores_qdrant_pgvector.txt",]},
    {"input": "Tell me about rerankers and evaluation", "suggested_files": ["04_rerankers_evaluation.txt",]},
    {"input": "Typical issues working with RAG", "suggested_files": ["05_troubleshooting_patterns.txt",]},
    {"input": "How to setup Enviroment?", "suggested_files": ["06_env_setup.txt",]},
    {"input": "Let's speak about llama cpp", "suggested_files": ["07_llama_cpp_setup.txt",]},
    {"input": "What metrics i need to evaluate for RAG?", "suggested_files": ["08_metrics_collection.txt",]},
    {"input": "How to deploy it with FastAPI?", "suggested_files": ["10_fastapi_deploy.txt",]},
]

for case in test_cases:
    _, scores = query.query(case["input"])

    print(f'scores for "{case["input"]}": {str(scores)}', end='\n\n')
    
    precision = len(set(scores.values()) & set(case["suggested_files"]))
    
    is_hit = precision >= 1
    
    print(f'Hit@5 for "{case["input"]}": {is_hit}')
    print(f'Precision@5 for "{case["input"]}": {(precision / 5):.1}', end='\n\n\n')