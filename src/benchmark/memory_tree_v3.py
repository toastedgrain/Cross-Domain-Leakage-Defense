"""
memory_tree_v3.py — Dynamic subdomain generation using Gemini API.

Instead of hardcoded subdomains, this version calls Gemini to generate
subdomain labels that are tailored to the actual memories of each user.
This reduces noise by ensuring subdomains reflect what is actually present
in the memory profile rather than generic categories.
"""

import os
import sys
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Fix: repo has a local 'datasets' folder that conflicts with the HuggingFace
# 'datasets' package required by sentence_transformers. Temporarily remove
# the src path, import sentence_transformers, then restore it.
_src_path = next((p for p in sys.path if p.endswith("src")), None)
if _src_path and _src_path in sys.path:
    sys.path.remove(_src_path)
from sentence_transformers import SentenceTransformer
if _src_path:
    sys.path.insert(0, _src_path)

# Use new google.genai package (old google.generativeai is deprecated)
try:
    from google import genai as _genai
    _USE_NEW_GENAI = True
except ImportError:
    import google.generativeai as _genai  # type: ignore
    _USE_NEW_GENAI = False


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
        _model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    return _model

def _get_domain_embeddings():
    global _domain_embeddings
    if _domain_embeddings is None:
        _domain_embeddings = _get_model().encode(DOMAINS)
    return _domain_embeddings


def _generate_subdomains_for_domain(domain: str, memories: list[str]) -> list[str]:
    """
    Call Gemini to generate 3-5 subdomain labels for a given domain
    based on the actual memories assigned to it.
    Falls back to a single generic label if Gemini call fails.
    """
    if not memories:
        return [domain]

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return [domain]

    try:
        memories_text = "\n".join(f"- {m}" for m in memories)
        prompt = f"""You are organizing personal memories into subcategories.

Domain: {domain}
Memories in this domain:
{memories_text}

Generate 3 to 5 short subdomain labels that best organize these specific memories.
Return ONLY a JSON array of strings. No explanation, no markdown, no extra text.
Example: ["medications and treatments", "medical appointments", "fitness habits"]"""

        if _USE_NEW_GENAI:
            client = _genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            text = response.text.strip()
        else:
            _genai.configure(api_key=api_key)
            gemini = _genai.GenerativeModel("gemini-2.0-flash")
            response = gemini.generate_content(prompt)
            text = response.text.strip()

        # Strip markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        subdomains = json.loads(text)
        if isinstance(subdomains, list) and len(subdomains) > 0:
            return [str(s) for s in subdomains]
        return [domain]

    except Exception:
        return [domain]


def filter_memories_for_query(memories: list[str], query: str) -> list[str]:
    if not memories:
        return []

    model = _get_model()
    domain_embeddings = _get_domain_embeddings()

    # Step 1: Assign each memory to a top-level domain
    memory_embeddings = model.encode(memories)
    memory_scores = cosine_similarity(memory_embeddings, domain_embeddings)

    domain_buckets: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
    for i, memory in enumerate(memories):
        best_domain = DOMAINS[int(np.argmax(memory_scores[i]))]
        domain_buckets[best_domain].append(memory)

    # Step 2: For each non-empty domain, call Gemini to generate dynamic subdomains
    dynamic_tree: dict[str, dict[str, list[str]]] = {}
    subdomain_embeddings_cache: dict[str, np.ndarray] = {}

    for domain, domain_memories in domain_buckets.items():
        if not domain_memories:
            dynamic_tree[domain] = {}
            continue

        subdomains = _generate_subdomains_for_domain(domain, domain_memories)
        subdomain_embs = model.encode(subdomains)
        subdomain_embeddings_cache[domain] = subdomain_embs

        sub_tree: dict[str, list[str]] = {sub: [] for sub in subdomains}
        domain_mem_embs = model.encode(domain_memories)

        for i, memory in enumerate(domain_memories):
            sub_scores = cosine_similarity(
                domain_mem_embs[i].reshape(1, -1), subdomain_embs
            )[0]
            best_sub = subdomains[int(np.argmax(sub_scores))]
            sub_tree[best_sub].append(memory)

        dynamic_tree[domain] = sub_tree

    # Step 3: Route query to top-level domain
    query_embedding = model.encode([query])
    query_domain_scores = cosine_similarity(query_embedding, domain_embeddings)[0]
    best_domain = DOMAINS[int(np.argmax(query_domain_scores))]

    # Step 4: Route query to best subdomain within that domain
    domain_sub_tree = dynamic_tree.get(best_domain, {})
    result = []

    if domain_sub_tree and best_domain in subdomain_embeddings_cache:
        subdomains = list(domain_sub_tree.keys())
        subdomain_embs = subdomain_embeddings_cache[best_domain]
        query_sub_scores = cosine_similarity(query_embedding, subdomain_embs)[0]
        best_sub = subdomains[int(np.argmax(query_sub_scores))]
        result = domain_sub_tree[best_sub]

    # Fallback 1: return all memories in domain
    if not result:
        result = [m for sub in domain_sub_tree.values() for m in sub]

    # Fallback 2: try second best domain
    if not result:
        second_domain = DOMAINS[int(np.argsort(query_domain_scores)[-2])]
        second_sub_tree = dynamic_tree.get(second_domain, {})
        result = [m for sub in second_sub_tree.values() for m in sub]

    return result if result else []
