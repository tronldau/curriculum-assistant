# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    OPENROUTER_MODEL_3B = "meta-llama/llama-3.2-3b-instruct"
    OPENROUTER_MODEL_1B = "meta-llama/llama-3.2-1b-instruct"
    
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'digit_curriculum')
    
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_MODEL = "llama-3.1-8b-instant"
    
    # Embedding Model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_SIZE = 384
    
    # RAG Settings
    TOP_K = 5

# Test if config loads correctly
if __name__ == "__main__":
    print("Testing Config...")
    print(f"MYSQL_HOST: {Config.MYSQL_HOST}")
    print(f"MYSQL_USER: {Config.MYSQL_USER}")
    print(f"MYSQL_DATABASE: {Config.MYSQL_DATABASE}")
    print(f"QDRANT_URL: {Config.QDRANT_URL[:50]}..." if Config.QDRANT_URL else "QDRANT_URL: None")
    print(f"GROQ_API_KEY: {'Set' if Config.GROQ_API_KEY else 'Not Set'}")