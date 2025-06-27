import requests
from bs4 import BeautifulSoup
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Web Crawling ---
def crawl_website(url: str, max_pages: int = 10) -> List[str]:
    visited = set()
    to_visit = [url]
    docs = []
    while to_visit and len(visited) < max_pages:
        current = to_visit.pop(0)
        if current in visited:
            continue
        try:
            resp = requests.get(current, timeout=5)
            soup = BeautifulSoup(resp.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            docs.append(text)
            visited.add(current)
            # Add new links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('http') and href not in visited:
                    to_visit.append(href)
        except Exception as e:
            print(f"Failed to crawl {current}: {e}")
    return docs

# --- Retrieval ---
def build_retrieval_index(docs: List[str], model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index, model, embeddings

def retrieve(query: str, index, model, docs: List[str], top_k: int = 5) -> List[str]:
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb).astype('float32'), top_k)
    return [docs[i] for i in I[0]]

# --- Evaluation ---
def evaluate_rag(answers: List[str], references: List[str]) -> float:
    # Simple exact match
    correct = 0
    for a, r in zip(answers, references):
        if a.strip().lower() == r.strip().lower():
            correct += 1
    return correct / len(answers) if answers else 0.0 