import pytest
from src.database import ChromaDBManager
from src.indexer import Indexer
from src.search_engine import SemanticSearchEngine

@pytest.fixture(scope="module")
def test_db():
    # Use an in-memory DB or temporary path for tests
    db_manager = ChromaDBManager(db_path="./test_chroma_db", collection_name="test_collection")
    yield db_manager
    # Teardown logic if needed

def test_indexer_and_search(test_db):
    indexer = Indexer(test_db)
    
    test_articles = [
        {"id": "test1", "titulo": "Python Backend", "contenido": "Python es genial para backend con FastAPI."},
        {"id": "test2", "titulo": "Frontend React", "contenido": "React se usa en el frontend para interfaces de usuario."}
    ]
    
    # Test indexing
    indexer.index_articles(test_articles)
    
    # Test search
    search_engine = SemanticSearchEngine(test_db)
    results = search_engine.search("backend python", n_resultados=1)
    
    assert len(results) > 0
    assert results[0].article.id == "test1"
    assert "Python" in results[0].article.titulo
    assert results[0].similarity_score > 0
