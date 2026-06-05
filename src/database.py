import os
import chromadb
from chromadb.utils import embedding_functions

class ChromaDBManager:
    """
    Manages connection and interaction with ChromaDB.
    Follows Single Responsibility Principle by handling only DB operations.
    """
    def __init__(self, db_path: str = "chroma_db", collection_name: str = "articulos"):
        # We ensure the persistence directory is absolute if needed, or relative to cwd.
        self.client = chromadb.PersistentClient(path=db_path)
        
        # We use SentenceTransformers open source model by default since it's free and local
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )

    def add_documents(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        """Adds documents to the collection. ChromaDB handles embeddings automatically."""
        # Note: If id already exists, it will be skipped or we can use upsert
        # We will use upsert to support incremental indexing and updates
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def query_documents(self, query: str, n_results: int = 3):
        """Queries the collection using the embedding function."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

    def get_all_documents(self):
        """Gets all documents for analysis."""
        return self.collection.get(include=['embeddings', 'metadatas', 'documents'])
