import os
from groq import Groq
from dotenv import load_dotenv
from src.models import SearchResult

class RAGGenerator:
    """
    Handles Retrieval-Augmented Generation using Groq.
    Generates a natural language response based on retrieved documents.
    """
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY no encontrada en variables de entorno")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama3-8b-8192" # Fast and capable Groq model

    def generate_answer(self, query: str, context_results: list[SearchResult]) -> str:
        if not context_results:
            return "No se encontró información relevante para responder a tu pregunta."

        # Construir el contexto
        context_texts = []
        for res in context_results:
            context_texts.append(f"Título: {res.article.titulo}\nContenido: {res.article.contenido}")
        
        context_str = "\n\n---\n\n".join(context_texts)

        prompt = f"""Eres un asistente experto en tecnología y programación. 
Tu tarea es responder a la pregunta del usuario utilizando ÚNICAMENTE la información proporcionada en el contexto. 
Si la respuesta no está en el contexto, di "No tengo suficiente información para responder a esto".

Contexto:
{context_str}

Pregunta del usuario: {query}

Respuesta:"""

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
                temperature=0.2, # Low temperature for more factual answers
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error al generar la respuesta con Groq: {e}"
