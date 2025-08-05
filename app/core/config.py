from pydantic_settings import BaseSettings
from typing import List
class Server(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
class Database(BaseSettings):
    URL: str = ""
    NAME: str = ""
class Settings(BaseSettings):
    SERVER: Server = Server()
    DATABASE: Database = Database()
    API_PREFIX: str = "/api/v1"
    TOGETHER_API_KEY: str
    LLM_MODEL: str = "meta-llama/Llama-3-70b-chat-hf_free"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_STORE_PATH: str = "vector_store.json"
    LANGFUSE_PUBLIC_KEY: str ="pk-lf-0c3b9ad9-3dee-4103-ae21-16c06cd1812f"
    LANGFUSE_SECRET_KEY: str ="sk-lf-8d21983d-ff0d-4151-b9c0-98e97c978cc9"
    OPENAI_API_KEY: str
    OPENAI_BASE_URL:str
    LLM_MODEL: str = "gpt-4o"
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    SIMILARITY_THRESHOLD: float = 0.7
    SUPPORT_TAGS: List[str] = [
        "پشتیبانی فنی",
        "فروش و قیمت‌گذاری",
        "مالی و صورتحساب",
        "حساب کاربری و ورود",
        "ارسال و تحویل",
        "پیشنهادات و انتقادات",
        "همکاری تجاری",
        "سوالات عمومی"
    ]
settings = Settings()
