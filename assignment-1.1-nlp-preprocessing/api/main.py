from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import nltk
import spacy
from typing import List, Dict, Any

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="NLP Preprocessing API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/tokenize")
async def tokenize(text_input: TextInput) -> Dict[str, List[str]]:
    """Tokenize input text using NLTK."""
    try:
        tokens = nltk.word_tokenize(text_input.text)
        return {"tokens": tokens}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stem")
async def stem(text_input: TextInput) -> Dict[str, List[str]]:
    """Apply stemming using NLTK's Porter Stemmer."""
    try:
        stemmer = nltk.PorterStemmer()
        tokens = nltk.word_tokenize(text_input.text)
        stems = [stemmer.stem(token) for token in tokens]
        return {"stems": stems}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/lemmatize")
async def lemmatize(text_input: TextInput) -> Dict[str, List[str]]:
    """Apply lemmatization using spaCy."""
    try:
        doc = nlp(text_input.text)
        lemmas = [token.lemma_ for token in doc]
        return {"lemmas": lemmas}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pos")
async def pos_tag(text_input: TextInput) -> Dict[str, List[Dict[str, str]]]:
    """Perform POS tagging using NLTK."""
    try:
        tokens = nltk.word_tokenize(text_input.text)
        pos_tags = nltk.pos_tag(tokens)
        return {"pos_tags": [{"word": word, "tag": tag} for word, tag in pos_tags]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ner")
async def named_entity_recognition(text_input: TextInput) -> Dict[str, List[Dict[str, Any]]]:
    """Perform Named Entity Recognition using spaCy."""
    try:
        doc = nlp(text_input.text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        return {"entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 