"""
Hybrid (Ensemble) Search RAG Implementation

Hybrid Search combines:
- Keyword-based search (BM25)
- Semantic search (vector embeddings)

Setup:
- LLM: Groq (https://console.groq.com/keys)
- Vector Store: ChromaDB
- Embedding Model: nomic-embed-text-v1.5
- Framework: LangChain
- Huggingface API Key: https://huggingface.co/settings/tokens
"""

# Install required libraries (run once):
# pip install -q -U Sentence-transformers==3.0.1 langchain==0.3.19 langchain-groq==0.2.4 \
#     langchain-chroma==0.2.2 langchain-community==0.3.18 langchain-huggingface==0.1.2 \
#     einops==0.8.1 rank_bm25==0.2.2

import getpass
import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# ── API Keys ──────────────────────────────────────────────────────────────────
os.environ["GROQ_API_KEY"] = getpass.getpass("Enter Groq API Key: ")
os.environ["HF_TOKEN"] = getpass.getpass("Enter Huggingface API Key: ")


# ── Step 1: Load and preprocess data ─────────────────────────────────────────
def load_and_process_data(url: str):
    """Load data from a URL and split into chunks."""
    loader = WebBaseLoader(url)
    data = loader.load()

    # Experiment with chunk_size and chunk_overlap for optimal results
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)

    return chunks


# ── Step 2: Create vector store and BM25 retriever ───────────────────────────
def create_retrievers(chunks):
    """Build an EnsembleRetriever combining BM25 and semantic (Chroma) search."""
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"trust_remote_code": True},
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5  # Number of documents to retrieve

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vectorstore.as_retriever(search_kwargs={"k": 5})],
        weights=[0.5, 0.5],
    )

    return ensemble_retriever


# ── Helper: Safe LLM call ─────────────────────────────────────────────────────
def safe_llm_call(prompt, **kwargs):
    """Invoke the LLM with error handling."""
    try:
        response = llm.invoke(prompt.format(**kwargs))
        return response.content if response else "No response generated."
    except Exception as e:
        print(f"Error in LLM call: {e}")
        return "An error occurred while generating the response."


# ── Step 3: Hybrid-search RAG ─────────────────────────────────────────────────
def hybrid_search_rag(query: str, ensemble_retriever, llm):
    """
    1. Hybrid Retrieval  – EnsembleRetriever (BM25 + semantic).
    2. Response Generation – answer the query using retrieved context.
    3. Hybrid Search Explanation – explain the benefit of the hybrid approach.
    """
    # Hybrid retrieval
    retrieved_docs = ensemble_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Generate response
    response_prompt = PromptTemplate.from_template(
        "You are an AI assistant tasked with answering questions based on the provided context. "
        "The context contains information retrieved using a hybrid search method combining keyword-based and semantic search. "
        "Please provide a comprehensive answer to the question, using the context when relevant "
        "and your general knowledge when necessary.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n"
        "Answer:"
    )
    final_answer = safe_llm_call(response_prompt, context=context, query=query)

    # Generate explanation of hybrid search process
    explanation_prompt = PromptTemplate.from_template(
        "Explain how the hybrid search process, combining keyword-based and semantic search, "
        "might have improved the retrieval of relevant information for answering the given query. "
        "Consider the potential benefits of this approach compared to using only one search method.\n\n"
        "Query: {query}\n"
        "Explanation:"
    )
    hybrid_search_explanation = safe_llm_call(explanation_prompt, query=query)

    return {
        "query": query,
        "final_answer": final_answer,
        "hybrid_search_explanation": hybrid_search_explanation,
        "retrieved_context": context,
    }


# ── Step 4: Initialise LLM, load data, build retriever ───────────────────────
llm = ChatGroq(model="llama3-8b-8192", temperature=0.5)

url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
chunks = load_and_process_data(url)
ensemble_retriever = create_retrievers(chunks)


# ── Step 5: Run Hybrid-Search RAG ─────────────────────────────────────────────
queries = [
    "What are the main applications of artificial intelligence in healthcare?",
    "Explain the concept of machine learning and its relationship to AI.",
    "Discuss the ethical implications of AI in decision-making processes.",
]

for query in queries:
    print(f"\nQuery: {query}")
    result = hybrid_search_rag(query, ensemble_retriever, llm)
    print("Final Answer:")
    print(result["final_answer"])
    print("\nHybrid Search Explanation:")
    print(result["hybrid_search_explanation"])
    print("\nRetrieved Context (first 300 characters):")
    print(result["retrieved_context"][:300] + "...")
