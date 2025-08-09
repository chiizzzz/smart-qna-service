
from pydantic import BaseModel
from typing import List

class QAResponse(BaseModel):
    tags_identified: List[str]
    final_answer: str
    retrieved_context_count: int

class AddKnowledgeResponse(BaseModel):
    """
    مدل پاسخی که پس از اضافه کردن دانش جدید به کاربر برگردانده می‌شود
    """
    message: str
    items_added_count: int
    total_items_in_db: int
# --- پایان بخش جدید ---