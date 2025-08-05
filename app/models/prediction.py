# مسیر فایل: app/models/prediction.py

from pydantic import BaseModel
from typing import List

# این کلاس بدون تغییر باقی می‌ماند
class QAResponse(BaseModel):
    tags_identified: List[str]
    final_answer: str
    retrieved_context_count: int

# --- بخش جدید برای تسک ۳ ---
class AddKnowledgeResponse(BaseModel):
    """
    مدل پاسخی که پس از اضافه کردن دانش جدید به کاربر برگردانده می‌شود
    """
    message: str
    items_added_count: int
    total_items_in_db: int
# --- پایان بخش جدید ---