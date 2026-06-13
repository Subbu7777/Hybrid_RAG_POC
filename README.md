# Hybrid Search RAG

A production-ready Retrieval-Augmented Generation (RAG) system combining **keyword-based search (BM25)** and **semantic search (vector embeddings)** for improved document retrieval and question answering.

## Features

✅ **Hybrid Retrieval** – Combines BM25 and semantic (ChromaDB) search via EnsembleRetriever  
✅ **LLM Integration** – Groq's free, open-source LLM endpoints (llama3-8b-8192)  
✅ **Semantic Embeddings** – nomic-embed-text-v1.5 for high-quality vector representations  
✅ **Error Handling** – Robust exception handling for safe LLM calls  
✅ **Context Awareness** – Generates explanations of how hybrid search improved retrieval  

## Quick Start

### 1. Install Dependencies

```bash
pip install -r hybrid_search_rag_requirements.txt
```

### 2. Set API Keys

Get your keys:
- **Groq API Key**: https://console.groq.com/keys
- **Huggingface Token**: https://huggingface.co/settings/tokens

The script will prompt for both keys at runtime.

### 3. Run

```bash
python hybrid_search_rag.py
```

## How It Works

### Step 1: Data Loading & Preprocessing
```python
url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
chunks = load_and_process_data(url)
```
- Loads web content via WebBaseLoader
- Splits into 500-char chunks with 50-char overlap

### Step 2: Dual Retrieval
```python
ensemble_retriever = create_retrievers(chunks)
```
- **BM25 Retriever**: Keyword-based search (50% weight)
- **ChromaDB Retriever**: Semantic/vector search (50% weight)
- Both weight equally for balanced results

### Step 3: RAG Response
```python
result = hybrid_search_rag(query, ensemble_retriever, llm)
```
Returns:
- `final_answer` – LLM response based on hybrid-retrieved context
- `hybrid_search_explanation` – Why this approach worked better
- `retrieved_context` – Context documents used

## Project Structure

```
hybrid_search_rag.py               # Main script
hybrid_search_rag_requirements.txt  # Dependencies
HYBRID_README.md                   # This file
```

## Key Components

| Component | Purpose |
|-----------|---------|
| `load_and_process_data()` | Fetch and chunk web content |
| `create_retrievers()` | Build BM25 + ChromaDB ensemble |
| `safe_llm_call()` | Error-safe LLM invocation |
| `hybrid_search_rag()` | Core RAG pipeline |

## Customization

### Adjust Chunking
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Increase for longer contexts
    chunk_overlap=100     # Increase for more overlap
)
```

### Change URL
```python
url = "https://your-website.com"
```

### Tune Ensemble Weights
```python
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vectorstore.as_retriever()],
    weights=[0.3, 0.7]  # More weight on semantic search
)
```

### Modify LLM Parameters
```python
llm = ChatGroq(
    model="llama3-8b-8192",  # or "mixtral-8x7b-32768"
    temperature=0.7          # Lower = deterministic, Higher = creative
)
```

## Requirements

- Python 3.8+
- `langchain` 0.3.19+
- `langchain-groq` 0.2.4+
- `langchain-chroma` 0.2.2+
- `langchain-huggingface` 0.1.2+
- `rank_bm25` 0.2.2+
- Internet connection (for embeddings & LLM)

## Example Output

```
Query: What are the main applications of artificial intelligence in healthcare?

Final Answer:
AI in healthcare is used for diagnostic imaging analysis, drug discovery, patient monitoring...

Hybrid Search Explanation:
The hybrid approach combined keyword matches for "AI", "healthcare", and "applications" 
with semantic understanding of medical context, retrieving more relevant documents than 
either method alone.

Retrieved Context (first 300 characters):
Artificial intelligence in medicine and healthcare is set to influence diagnostics, 
patient management and therapy development. Hospitals and medical offices around the world...
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r hybrid_search_rag_requirements.txt` |
| API key errors | Ensure keys are valid and have correct permissions |
| Slow embeddings | First run downloads the 200MB+ model; subsequent runs use cache |
| Low relevance results | Increase `k` in retrievers or adjust chunk size |

## Performance Tips

- **Batch queries** – Process multiple queries sequentially to reuse the ensemble retriever
- **Cache embeddings** – ChromaDB stores embeddings locally; subsequent runs are faster
- **Optimize chunk size** – Smaller chunks for specific queries, larger for comprehensive answers

## License

MIT

## References

- [LangChain Documentation](https://python.langchain.com/)
- [Groq API](https://groq.com/)
- [ChromaDB](https://www.trychroma.com/)
- [nomic-embed-text](https://www.nomic.ai/blog/posts/nomic-embed-text-v1)
