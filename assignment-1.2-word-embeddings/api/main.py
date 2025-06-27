from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import gensim.downloader as api
from typing import List, Dict, Any
import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')

app = FastAPI(title="Word Embeddings API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load GloVe model
try:
    glove_model = api.load("glove-wiki-gigaword-100")
except Exception as e:
    print(f"Error loading GloVe model: {e}")
    glove_model = None

class TextInput(BaseModel):
    text: str

class WordInput(BaseModel):
    word: str
    n_neighbors: int = 5

@app.post("/tfidf")
async def get_tfidf_embeddings(text_input: TextInput) -> Dict[str, Any]:
    """Generate TF-IDF embeddings for input text."""
    try:
        # Tokenize and preprocess
        tokens = word_tokenize(text_input.text.lower())
        text = ' '.join(tokens)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text])
        
        # Get feature names and their scores
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Create word-score pairs
        word_scores = [
            {"word": word, "score": float(score)}
            for word, score in zip(feature_names, scores)
            if score > 0
        ]
        
        return {
            "embeddings": word_scores,
            "dimension": len(feature_names)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/glove/vector")
async def get_glove_vector(word_input: WordInput) -> Dict[str, Any]:
    """Get GloVe vector for a word."""
    try:
        if glove_model is None:
            raise HTTPException(status_code=500, detail="GloVe model not loaded")
        
        word = word_input.word.lower()
        if word not in glove_model:
            raise HTTPException(status_code=404, detail=f"Word '{word}' not found in vocabulary")
        
        vector = glove_model[word].tolist()
        return {
            "word": word,
            "vector": vector,
            "dimension": len(vector)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/glove/neighbors")
async def get_glove_neighbors(word_input: WordInput) -> Dict[str, List[Dict[str, Any]]]:
    """Get nearest neighbors for a word using GloVe embeddings."""
    try:
        if glove_model is None:
            raise HTTPException(status_code=500, detail="GloVe model not loaded")
        
        word = word_input.word.lower()
        if word not in glove_model:
            raise HTTPException(status_code=404, detail=f"Word '{word}' not found in vocabulary")
        
        neighbors = glove_model.most_similar(word, topn=word_input.n_neighbors)
        return {
            "word": word,
            "neighbors": [
                {"word": neighbor, "similarity": float(similarity)}
                for neighbor, similarity in neighbors
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/glove/visualize")
async def visualize_glove_embeddings(words: List[str]) -> Dict[str, Any]:
    """Generate 2D visualization coordinates for multiple words using t-SNE."""
    try:
        if glove_model is None:
            raise HTTPException(status_code=500, detail="GloVe model not loaded")
        
        # Get vectors for all words
        vectors = []
        valid_words = []
        for word in words:
            word = word.lower()
            if word in glove_model:
                vectors.append(glove_model[word])
                valid_words.append(word)
        
        if not vectors:
            raise HTTPException(status_code=404, detail="No valid words found in vocabulary")
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        coords = tsne.fit_transform(vectors)
        
        # Create visualization data
        visualization = [
            {
                "word": word,
                "x": float(x),
                "y": float(y)
            }
            for word, (x, y) in zip(valid_words, coords)
        ]
        
        return {
            "visualization": visualization,
            "dimension": 2
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 