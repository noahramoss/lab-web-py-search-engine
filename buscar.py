import sys
from src.database import ChromaDBManager
from src.search_engine import SemanticSearchEngine
from src.rag_generator import RAGGenerator

def buscar(query: str, n_resultados: int = 3):
    db_manager = ChromaDBManager()
    search_engine = SemanticSearchEngine(db_manager)
    
    print(f"\nBuscando: '{query}'")
    print("-" * 50)
    
    results = search_engine.search(query, n_resultados)
    
    if not results:
        print("No se encontraron resultados.")
        return
        
    for idx, res in enumerate(results, 1):
        print(f"Resultado {idx}: {res.article.titulo} (Score: {res.similarity_score:.4f})")
        print(f"Contenido: {res.article.contenido}\n")
        
    print("-" * 50)
    print("Generando respuesta con Groq...")
    
    try:
        rag = RAGGenerator()
        respuesta = rag.generate_answer(query, results)
        print(f"\nRespuesta IA: {respuesta}")
    except Exception as e:
        print(f"No se pudo usar Groq: {e}")

if __name__ == "__main__":
    queries_prueba = [
        "¿cómo hacer una API en Python?",
        "diferencias entre frameworks de frontend",
        "cómo funciona la autenticación en aplicaciones web",
        "herramientas para trabajar con modelos de lenguaje"
    ]
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        buscar(query)
    else:
        print("Ejecutando pruebas predeterminadas:\n")
        for q in queries_prueba:
            buscar(q)
