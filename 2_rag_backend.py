# rag_backend.py
import warnings
import numpy as np

# Suppress all warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
np.seterr(all='ignore')

from typing import List
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
try:
    from ddgs import DDGS
except ImportError:
    # Fallback to old package name
    from duckduckgo_search import DDGS

# 1) Load corpus from Hugging Face
def load_texts() -> list[str]:
    ds = load_dataset("m-ric/huggingface_doc", split="train[:500]")  # small subset
    return [row["text"] for row in ds]

# 2) Chunk documents
def build_chunks(texts: list[str]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    docs = splitter.create_documents(texts)
    return docs

# 3) Embeddings
from langchain_core.embeddings import Embeddings

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        import numpy as np
        self.np = np
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with normalization to prevent divide by zero warnings."""
        import numpy as np
        
        if not texts:
            return []
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return []
        
        try:
            embeddings = self.model.encode(valid_texts, normalize_embeddings=True, convert_to_numpy=True)
            embeddings = np.array(embeddings, dtype=np.float32)
            
            # Double-check normalization to ensure no divide by zero issues
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Replace zero or very small norms with 1 to avoid divide by zero
            norms = np.where(norms < 1e-8, 1.0, norms)
            embeddings = embeddings / norms
            
            # Ensure no NaN or Inf values
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
            # Renormalize after cleaning
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            embeddings = embeddings / norms
            
            return embeddings.tolist()
        except Exception as e:
            # Silently handle embedding errors - return empty list
            return []
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with normalization."""
        import numpy as np
        
        if not text or not text.strip():
            # Return a small random normalized vector instead of zeros
            # This avoids divide by zero in Qdrant's dot product
            zero_vec = np.random.normal(0, 0.01, 384).astype(np.float32)
            norm = np.linalg.norm(zero_vec)
            if norm > 1e-8:  # Avoid division by very small numbers
                zero_vec = zero_vec / norm
            return zero_vec.tolist()
        
        try:
            embedding = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
            embedding = np.array(embedding, dtype=np.float32)
            
            # Force normalization and handle edge cases
            norm = np.linalg.norm(embedding)
            if norm < 1e-8:  # Very small or zero norm
                # Use a small random vector as fallback
                embedding = np.random.normal(0, 0.01, 384).astype(np.float32)
                norm = np.linalg.norm(embedding)
            
            embedding = embedding / norm
            
            # Ensure no NaN or Inf values
            embedding = np.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)
            # Renormalize after cleaning
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
            
            return embedding.tolist()
        except Exception as e:
            # Fallback: return a normalized random vector
            fallback = np.random.normal(0, 0.01, 384).astype(np.float32)
            norm = np.linalg.norm(fallback)
            if norm > 1e-8:
                fallback = fallback / norm
            return fallback.tolist()

embedding_model = SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts: list[str]) -> list[list[float]]:
    return embedding_model.embed_documents(texts)

# 4) Vector store + retriever
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

texts = load_texts()
docs = build_chunks(texts)
contents = [d.page_content for d in docs]
metadatas = [d.metadata for d in docs]

# Create Qdrant client and collection manually to avoid init_from parameter issue
collection_name = "hf_docs_rag"

# Initialize vectorstore using from_documents which should work better
try:
    vectorstore = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embedding_model,
        location=":memory:",
        collection_name=collection_name,
        force_recreate=True,
    )
except Exception as e:
    # Fallback: manually create client and collection, then add documents
    import uuid
    client = QdrantClient(location=":memory:")
    
    # Delete collection if it exists
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    # Create collection
    vector_size = 384
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    
    # Add documents manually
    embeddings = embedding_model.embed_documents(contents)
    points = []
    for i, (text, embedding, metadata) in enumerate(zip(contents, embeddings, metadatas)):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": embedding,
            "payload": {"page_content": text, **metadata}
        })
    
    # Batch upload points
    from qdrant_client.models import PointStruct
    client.upsert(
        collection_name=collection_name,
        points=[PointStruct(**p) for p in points]
    )
    
    # Create vectorstore wrapper
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )

# k=2 for faster MULTIHOP performance
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

def retrieve_docs(query: str) -> List[Document]:
    return retriever.invoke(query)

def retrieve_context(query: str) -> str:
    docs = retrieve_docs(query)
    return "\n\n".join(d.page_content for d in docs)

# 5) Web search fallback (short snippets)
def web_search_snippets(query: str, num: int = 3) -> List[str]:
    """
    Web search with error handling and reduced results for speed.
    Reduced to 3 results for faster MULTIHOP performance.
    """
    results = []
    try:
        # Use reduced results for speed
        with DDGS() as ddg:
            count = 0
            for r in ddg.text(query, max_results=num):
                if count >= num:
                    break
                results.append(f"{r.get('title','')}: {r.get('body','')}")
                count += 1
    except Exception as e:
        # Return empty rather than hanging
        return [f"Web search unavailable: {str(e)[:30]}"]
    
    return results if results else [f"No web results for: {query[:40]}"]

def web_search_context(query: str) -> str:
    """
    Get web search context with reduced results for speed.
    """
    # Reduce to 3 results for faster search
    snippets = web_search_snippets(query, num=3)
    return "\n\n".join(snippets)
