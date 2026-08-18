from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Path to the directory where the data files will be stored
    base_dir: Path = Path(__file__).parent
    data_dir: Path = base_dir / "data"

    db_path: Path = data_dir / "chroma_db"
    pdf_path: Path = data_dir / "document.pdf"

    # Settings for the LLM
    llm_model: str = "qwen2.5:3b"
    embeddings_model: str = "nomic-embed-text"

    # Redis settings
    redis_url: str = "redis://localhost:6379/0"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    search_kwargs: int = 3

    # Settings for the RAG
    max_history: int = 10
    contextualize_system_prompt: str = (
        "Data la cronologia della chat e l'ultima domanda dell'utente "
        "che potrebbe fare riferimento al contesto nella cronologia della chat, "
        "formula una domanda indipendente che possa essere compresa "
        "senza la cronologia. NON rispondere alla domanda, riformulala solo se necessario."
    )
    contextualize_prompt: str = (
        "Sei un assistente utile ed esperto. Usa SOLO il seguente contesto per rispondere alla domanda. "
        "Se non conosci la risposta in base al contesto, dì semplicemente che non lo sai, non inventare nulla.\n\n"
        "Contesto: {context}"
    )

    class Config:
        env_file = ".env"

settings = Settings()