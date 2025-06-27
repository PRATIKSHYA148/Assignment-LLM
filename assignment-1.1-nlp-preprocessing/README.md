# NLP Preprocessing Assignment

This assignment implements various NLP preprocessing techniques using NLTK and spaCy, with a FastAPI backend and interactive web interface.

## Features

- Text tokenization
- Stemming (Porter Stemmer)
- Lemmatization
- Part-of-Speech (POS) tagging
- Named Entity Recognition (NER)

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data and spaCy model:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
python -m spacy download en_core_web_sm
```

## Running the Application

1. Start the FastAPI backend:
```bash
cd api
uvicorn main:app --reload
```

2. Open the web interface:
- Open `webapp/index.html` in your web browser
- Or serve it using a simple HTTP server:
```bash
cd webapp
python -m http.server 8080
```
Then visit `http://localhost:8080`

## API Endpoints

- `POST /tokenize`: Tokenize input text
- `POST /stem`: Apply stemming
- `POST /lemmatize`: Apply lemmatization
- `POST /pos`: Perform POS tagging
- `POST /ner`: Perform Named Entity Recognition

All endpoints accept JSON input in the format:
```json
{
    "text": "Your input text here"
}
```

## Notebooks

The `notebooks` directory contains Jupyter notebooks demonstrating:
- Comparison of stemming vs lemmatization
- Examples of different preprocessing techniques
- Analysis of preprocessing results

## Technologies Used

- Backend: FastAPI
- NLP Libraries: NLTK, spaCy
- Frontend: HTML, JavaScript, Bootstrap 