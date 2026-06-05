from pydantic import BaseModel, Field

class Article(BaseModel):
    id: str = Field(..., description="Unique identifier of the article")
    titulo: str = Field(..., description="Title of the article")
    contenido: str = Field(..., description="Content of the article")

class SearchResult(BaseModel):
    article: Article
    similarity_score: float = Field(..., description="Cosine distance / similarity score")

class RAGResponse(BaseModel):
    query: str
    results: list[SearchResult]
    answer: str = Field(..., description="Generated answer from Groq based on context")
