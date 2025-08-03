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
    LLM_MODEL: str = "meta-llama/Llama-3-70b-chat-hf"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_STORE_PATH: str = "vector_store.json"
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