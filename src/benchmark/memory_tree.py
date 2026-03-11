from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DOMAINS = [
    "personal",
    "education",
    "employment",
    "finance",
    "housing",
    "legal",
    "mental health",
    "schedule",
    "health"
]

_model = None
_domain_embeddings = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def _get_domain_embeddings():
    global _domain_embeddings
    if _domain_embeddings is None:
        _domain_embeddings = _get_model().encode(DOMAINS)
    return _domain_embeddings

def filter_memories_for_query(memories, query):
    if not memories:
        return []
    model = _get_model()
    domain_embeddings = _get_domain_embeddings()
    memory_embeddings = model.encode(memories)
    memory_scores = cosine_similarity(memory_embeddings, domain_embeddings)
    tree = {domain: [] for domain in DOMAINS}
    for i, memory in enumerate(memories):
        best_domain = DOMAINS[int(np.argmax(memory_scores[i]))]
        tree[best_domain].append(memory)
    query_embedding = model.encode([query])
    query_scores = cosine_similarity(query_embedding, domain_embeddings)[0]
    best_domain = DOMAINS[int(np.argmax(query_scores))]
    result = tree[best_domain]
    if not result:
        second_idx = int(np.argsort(query_scores)[-2])
        result = tree[DOMAINS[second_idx]]
    return result if result else []
