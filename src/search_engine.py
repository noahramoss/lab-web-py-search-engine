from src.database import ChromaDBManager
from src.models import SearchResult, Article

class SemanticSearchEngine:
    """
    Handles semantic search using embeddings.
    """
    def __init__(self, db_manager: ChromaDBManager):
        self.db = db_manager

    def search(self, query: str, n_resultados: int = 3) -> list[SearchResult]:
        """
        Searches for the most similar documents.
        ChromaDB returns distance (0 is exact match, larger is less similar).
        We calculate a similarity score from it.
        """
        results = self.db.query_documents(query=query, n_results=n_resultados)
        
        search_results = []
        if not results["ids"] or not results["ids"][0]:
            return search_results
            
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            
            # Chroma with cosine space returns distance = 1 - cosine_similarity
            # Similarity score = 1 - distance
            similarity = max(0.0, 1.0 - distance)
            
            article = Article(
                id=doc_id,
                titulo=metadata.get("titulo", "Sin Título"),
                contenido=metadata.get("contenido", "")
            )
            
            search_results.append(SearchResult(
                article=article,
                similarity_score=similarity
            ))
            
        return search_results
