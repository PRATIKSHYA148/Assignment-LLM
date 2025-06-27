# Assignment 3.1: Retrieval-Augmented Generation (RAG)

This assignment implements a basic Retrieval-Augmented Generation (RAG) pipeline. The system crawls a given website, builds a retrieval index, and answers questions using retrieved context and a language model.

## Objectives
- Crawl a website and build a document store.
- Implement a retriever to fetch relevant documents for a query.
- Use a language model to generate answers based on retrieved context.
- Evaluate the RAG system on a question-answering dataset (e.g., Natural Questions or TriviaQA).

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebook: `rag_pipeline.ipynb`
3. Place datasets in the `data/` folder.

## Directory Structure
- `rag_pipeline.ipynb`: Main implementation notebook
- `requirements.txt`: Python dependencies
- `data/`: Datasets and crawled documents
- `utils.py`: Helper functions (crawling, retrieval, evaluation)

---

Update this README as you progress with the assignment. 