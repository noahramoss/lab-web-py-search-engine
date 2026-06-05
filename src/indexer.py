import tiktoken
from src.database import ChromaDBManager
from src.models import Article

class Indexer:
    """
    Handles the indexing of articles into the database.
    """
    def __init__(self, db_manager: ChromaDBManager):
        self.db = db_manager
        # Tiktoken encoding used by OpenAI just for token estimation
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback
            self.encoding = None

    def estimate_tokens(self, text: str) -> int:
        if not self.encoding:
            return len(text.split()) # Rough estimate
        return len(self.encoding.encode(text))

    def index_articles(self, articles_data: list[dict]):
        """
        Indexes a list of articles. Creates embeddings and stores in DB.
        """
        total_tokens = 0
        ids = []
        documents = []
        metadatas = []

        for item in articles_data:
            article = Article(**item)
            
            # Bonus: Index by title + content for better semantic search
            combined_text = f"Título: {article.titulo}\nContenido: {article.contenido}"
            
            ids.append(article.id)
            documents.append(combined_text)
            
            # Store title and content in metadata
            metadatas.append({
                "titulo": article.titulo,
                "contenido": article.contenido
            })
            
            total_tokens += self.estimate_tokens(combined_text)

        # Upsert documents
        self.db.add_documents(ids=ids, documents=documents, metadatas=metadatas)

        print(f"Indexación completada. Se han indexado {len(ids)} artículos.")
        print(f"Total de tokens procesados (estimado): {total_tokens}")
        # Note: Since we are using local models, the cost is $0.00
        print(f"Coste estimado: $0.00 (Usando embeddings locales open-source)")
