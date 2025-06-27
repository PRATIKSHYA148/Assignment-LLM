# Word Embeddings Assignment

This assignment implements word embeddings using TF-IDF and GloVe, with an interactive web interface for exploring word vectors and their relationships.

## Features

- TF-IDF embedding generation for text
- GloVe word vector exploration
- Nearest neighbor analysis
- Interactive 2D visualization using t-SNE
- Real-time word similarity exploration

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

3. Download required NLTK data:
```bash
python -c "import nltk; nltk.download('punkt')"
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

### TF-IDF Endpoints
- `POST /tfidf`: Generate TF-IDF embeddings for input text
  ```json
  {
      "text": "Your input text here"
  }
  ```

### GloVe Endpoints
- `POST /glove/vector`: Get GloVe vector for a word
  ```json
  {
      "word": "example",
      "n_neighbors": 5
  }
  ```
- `POST /glove/neighbors`: Get nearest neighbors for a word
  ```json
  {
      "word": "example",
      "n_neighbors": 5
  }
  ```
- `POST /glove/visualize`: Generate 2D visualization for multiple words
  ```json
  ["word1", "word2", "word3", ...]
  ```

## Notebooks

The `notebooks` directory contains Jupyter notebooks demonstrating:
- Word vector exploration and analysis
- Comparison of different embedding methods
- Visualization techniques for word embeddings

## Technologies Used

- Backend: FastAPI
- Word Embeddings: Gensim (GloVe), scikit-learn (TF-IDF)
- Visualization: Plotly.js
- Frontend: HTML, JavaScript, Bootstrap

## Notes

- The GloVe model used is "glove-wiki-gigaword-100" (100-dimensional vectors)
- t-SNE is used for dimensionality reduction in visualizations
- The web interface provides real-time exploration of word relationships 