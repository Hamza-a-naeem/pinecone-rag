# backend/src/config.py

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # ==================== Application Settings ====================
    APP_NAME: str = "Company Docs Copilot"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"  # development, staging, production
    
    # ==================== API Configuration ====================
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]
    
    # ==================== LLM Provider Configuration ====================
    # Options: "openai", "anthropic", "ollama"
    LLM_PROVIDER: str = "openai"
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    OPENAI_ORG_ID: str = ""  # Optional
    
    # Anthropic Configuration (if using Claude)
    ANTHROPIC_API_KEY: str = ""
    
    # Ollama Configuration (if using local models)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # ==================== Model Configuration ====================
    # OpenAI models: gpt-4o, gpt-4-turbo, gpt-3.5-turbo
    # Anthropic models: claude-3-5-sonnet-20241022, claude-3-opus-20240229
    # Ollama models: llama3.1, mistral, codellama
    LLM_MODEL: str = "gpt-4o"
    
    # Embedding models
    # OpenAI: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
    # For local: sentence-transformers/all-MiniLM-L6-v2
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536  # 1536 for OpenAI small, 3072 for large
    
    # Generation parameters
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.1
    TOP_P: float = 1.0
    
    # ==================== Vector Database Configuration ====================
    # Options: "pinecone", "weaviate", "pgvector"
    VECTOR_DB: str = "pinecone"
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = "gcp-starter"  # or us-east-1-aws, etc.
    PINECONE_INDEX_NAME: str = "company-docs"
    
    # Weaviate Configuration
    WEAVIATE_URL: str = "http://localhost:8080"
    WEAVIATE_API_KEY: str = ""
    WEAVIATE_CLASS_NAME: str = "Document"
    
    # PostgreSQL + pgvector Configuration
    DATABASE_URL: str = "postgresql://copilot:copilot@localhost:5432/copilot"
    PGVECTOR_TABLE_NAME: str = "documents"
    
    # ==================== Document Processing Configuration ====================
    # Text chunking strategy
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Supported file types
    SUPPORTED_EXTENSIONS: List[str] = [".txt", ".md", ".pdf", ".docx", ".doc"]
    
    # Maximum file size (in MB)
    MAX_FILE_SIZE_MB: int = 10
    
    # ==================== Retrieval Configuration ====================
    # Number of documents to retrieve
    TOP_K_RESULTS: int = 5
    
    # Minimum similarity score threshold (0.0 to 1.0)
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Reranking (if using)
    USE_RERANKER: bool = False
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # ==================== Redis Configuration (for caching) ====================
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_ENABLED: bool = False
    CACHE_TTL: int = 3600  # 1 hour in seconds
    
    # ==================== Rate Limiting ====================
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 20
    RATE_LIMIT_PER_HOUR: int = 100
    
    # ==================== Logging Configuration ====================
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT: str = "json"  # json or text
    
    # ==================== Monitoring & Metrics ====================
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # ==================== Security ====================
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ==================== Feature Flags ====================
    ENABLE_CHAT_HISTORY: bool = True
    ENABLE_SOURCE_CITATIONS: bool = True
    ENABLE_FEEDBACK: bool = True
    ENABLE_STREAMING: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Using lru_cache ensures we only create one Settings instance.
    """
    return Settings()


# Global settings instance
settings = get_settings()


# Validation functions
def validate_settings():
    """Validate critical settings on startup."""
    errors = []
    
    if settings.LLM_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required when using OpenAI")
    
    if settings.LLM_PROVIDER == "anthropic" and not settings.ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is required when using Anthropic")
    
    if settings.VECTOR_DB == "pinecone" and not settings.PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY is required when using Pinecone")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True