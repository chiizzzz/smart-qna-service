# مسیر فایل: app/models/payload.py

from pydantic import BaseModel
from typing import List, Optional

# --- این بخش برای تسک ۱ و ۲ است (بدون تغییر) ---
class QARequest(BaseModel):
    query: str

# --- این بخش جدید برای تسک ۳ و ۴ است ---

class SingleQA(BaseModel):
    """مدلی برای یک جفت پرسش و پاسخ."""
    question: str
    answer: str

class QAList(BaseModel):
    """مدلی برای دریافت لیستی از پرسش و پاسخ‌ها."""
    data: List[SingleQA]

class DeleteRequest(BaseModel):
    """مدلی برای درخواست حذف آیتم‌ها بر اساس ID."""
    ids: List[str]

class OperatorAnswerPayload(BaseModel):
    """
    مدلی برای دریافت پاسخ ثبت شده توسط اپراتور برای یک سوال بی‌پاسخ.
    """
    question: str
    answer: str