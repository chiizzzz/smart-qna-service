from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    API_PREFIX: str = "/api/v1"

    # --- مدل زبانی (هر API سازگار با OpenAI) ---
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: Optional[str] = None
    LLM_MODEL: str = "gpt-4o"
    LLM_MAX_TOKENS: int = 512

    # --- امبدینگ و پایگاه دانش ---
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    CHROMA_DB_PATH: str = "./chroma_db_store"
    # شباهت کسینوسی (بین -1 و 1). مقدار 0.75 معادل آستانه 0.5 قبلی روی فاصله L2 است.
    SIMILARITY_THRESHOLD: float = 0.75
    TOP_K: int = 3

    # --- کش پاسخ‌ها برای حلقه بازخورد ---
    CACHE_MAX_SIZE: int = 1000

    # --- امنیت مسیرهای ادمین ---
    # اگر خالی باشد، تمام مسیرهای ادمین غیرفعال (503) هستند.
    ADMIN_API_KEY: Optional[str] = None

    # --- Langfuse (اختیاری؛ فقط از طریق .env مقداردهی شود) ---
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None

    SUPPORT_TAGS: List[str] = [
        "پشتیبانی فنی",
        "فروش و قیمت‌گذاری",
        "مالی و صورتحساب",
        "حساب کاربری و ورود",
        "ارسال و تحویل",
        "پیشنهادات و انتقادات",
        "همکاری تجاری"
    ]


settings = Settings()
