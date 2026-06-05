from fastapi import FastAPI, Query, HTTPException
from src.database import ChromaDBManager
from src.search_engine import SemanticSearchEngine
from src.rag_generator import RAGGenerator
from src.models import RAGResponse, SearchResult

app = FastAPI(
    title="Motor de Búsqueda Semántica API",
    description="API para buscar artículos usando embeddings y RAG con Groq"
)

# Inicializar dependencias
db_manager = ChromaDBManager()
search_engine = SemanticSearchEngine(db_manager)
try:
    rag_generator = RAGGenerator()
except Exception as e:
    print(f"Warning: RAG no disponible. {e}")
    rag_generator = None

@app.get("/")
def read_root():
    return {"message": "Bienvenido al Motor de Búsqueda Semántica"}

@app.get("/buscar", response_model=list[SearchResult])
def buscar(q: str = Query(..., description="Consulta a buscar"), n: int = 3):
    """
    Endpoint para buscar documentos similares en ChromaDB usando embeddings.
    """
    try:
        results = search_engine.search(q, n_resultados=n)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag", response_model=RAGResponse)
def rag_search(q: str = Query(..., description="Consulta para responder mediante RAG")):
    """
    Endpoint que usa RAG (Retrieval-Augmented Generation).
    1. Busca los documentos relevantes.
    2. Usa Groq para redactar una respuesta usando esos documentos como contexto.
    """
    if not rag_generator:
        raise HTTPException(status_code=503, detail="RAG no está disponible. Comprueba tu API Key de Groq.")
        
    try:
        # Recuperar
        results = search_engine.search(q, n_resultados=3)
        # Generar
        answer = rag_generator.generate_answer(q, results)
        
        return RAGResponse(
            query=q,
            results=results,
            answer=answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
