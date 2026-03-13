from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DOMAIN_SUBDOMAIN_MAP = {
    "health": [
        "physical health and medical conditions",
        "mental health and psychological wellbeing",
        "treatments and medications",
        "fitness and exercise",
        "therapy and counseling"
    ],
    "identity": [
        "nationality and ethnicity",
        "religion and spiritual beliefs",
        "gender identity and sexuality",
        "personal values and beliefs"
    ],
    "social": [
        "friendships and friend groups",
        "family relationships",
        "workplace colleagues and professional relationships",
        "acquaintances and community"
    ],
    "romantic": [
        "dating and romantic interests",
        "partners and marriage",
        "attraction and intimacy",
        "breakups and relationship history"
    ],
    "personal": [
        "hobbies and interests",
        "lifestyle choices and preferences",
        "personality traits",
        "daily habits and routines"
    ],
    "education": [
        "schooling and degrees",
        "courses and certifications",
        "academic history and performance",
        "learning experiences and tutoring"
    ],
    "employment": [
        "current job and work history",
        "workplace experiences and culture",
        "professional skills and expertise",
        "colleagues and managers"
    ],
    "finance": [
        "income and salary",
        "savings and investments",
        "expenses and debt",
        "banking and taxes"
    ],
    "housing": [
        "home and residence location",
        "living situation and roommates",
        "rent and mortgage",
        "neighbors and neighborhood"
    ],
    "legal": [
        "legal issues and disputes",
        "contracts and agreements",
        "rights and official documents",
        "criminal record and court matters"
    ],
    "schedule": [
        "appointments and meetings",
        "daily routines and habits",
        "recurring events and commitments",
        "time-based plans"
    ]
}

DOMAINS = list(DOMAIN_SUBDOMAIN_MAP.keys())

_model = None
_domain_embeddings = None
_subdomain_embeddings = {}


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    return _model


def _get_domain_embeddings():
    global _domain_embeddings
    if _domain_embeddings is None:
        _domain_embeddings = _get_model().encode(DOMAINS)
    return _domain_embeddings


def _get_subdomain_embeddings(domain):
    global _subdomain_embeddings
    if domain not in _subdomain_embeddings:
        _subdomain_embeddings[domain] = _get_model().encode(DOMAIN_SUBDOMAIN_MAP[domain])
    return _subdomain_embeddings[domain]


def filter_memories_for_query(memories, query):
    if not memories:
        return []

    model = _get_model()
    domain_embeddings = _get_domain_embeddings()

    # Step 1: Encode all memories
    memory_embeddings = model.encode(memories)
    memory_scores = cosine_similarity(memory_embeddings, domain_embeddings)

    # Step 2: Build two-level tree — domain > subdomain > memories
    tree = {domain: {sub: [] for sub in DOMAIN_SUBDOMAIN_MAP[domain]} for domain in DOMAINS}

    for i, memory in enumerate(memories):
        best_domain = DOMAINS[int(np.argmax(memory_scores[i]))]
        subdomain_embeddings = _get_subdomain_embeddings(best_domain)
        sub_scores = cosine_similarity(memory_embeddings[i].reshape(1, -1), subdomain_embeddings)[0]
        best_sub = DOMAIN_SUBDOMAIN_MAP[best_domain][int(np.argmax(sub_scores))]
        tree[best_domain][best_sub].append(memory)

    # Step 3: Route query to top-level domain
    query_embedding = model.encode([query])
    query_domain_scores = cosine_similarity(query_embedding, domain_embeddings)[0]
    best_domain = DOMAINS[int(np.argmax(query_domain_scores))]

    # Step 4: Route query to subdomain within that domain
    subdomain_embeddings = _get_subdomain_embeddings(best_domain)
    query_sub_scores = cosine_similarity(query_embedding, subdomain_embeddings)[0]
    best_sub = DOMAIN_SUBDOMAIN_MAP[best_domain][int(np.argmax(query_sub_scores))]

    result = tree[best_domain][best_sub]

    # Fallback 1: return all memories in the domain if subdomain is empty
    if not result:
        result = [m for sub in tree[best_domain].values() for m in sub]

    # Fallback 2: try second best domain
    if not result:
        second_domain = DOMAINS[int(np.argsort(query_domain_scores)[-2])]
        result = [m for sub in tree[second_domain].values() for m in sub]

    return result if result else []
