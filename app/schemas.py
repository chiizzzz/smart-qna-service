from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import uuid
from datetime import datetime

class QAResponse(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tags_identified: List[str]
    final_answer: str
    retrieved_context_count: int
    original_question: str

class FeedbackRequest(BaseModel):
    """
    مدلی برای دریافت بازخورد کاربر در مورد یک پاسخ.
    """
    session_id: str
    is_correct: bool = Field(..., description="آیا پاسخ دریافت شده صحیح و مفید بود؟")


class PendingTicket(BaseModel):
    """مدل یک تیکت در حال انتظار در فایل JSON."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    timestamp: datetime = Field(default_factory=datetime.now)
    # --- فیلدهای جدید ---
    bot_answer: Optional[str] = None
    ticket_source: Literal["unanswered", "negative_feedback"] = "unanswered"

class FeedbackResponse(BaseModel):
    """
    پاسخی که پس از ثبت بازخورد به کاربر داده می‌شود.
    """
    status: str
    message: str
class AddKnowledgeResponse(BaseModel):
    """
    مدل پاسخی که پس از اضافه کردن دانش جدید به کاربر برگردانده می‌شود
    """
    message: str
    items_added_count: int
    total_items_in_db: int

class QARequest(BaseModel):
    query: str

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

class TicketCreationResponse(BaseModel):
    """پاسخی که هنگام ایجاد تیکت جدید برای کاربر ارسال می‌شود."""
    status: Literal["ticket_created"] = "ticket_created"
    message: str
    question_id: str

class PendingTicket(BaseModel):
    """مدل یک تیکت در حال انتظار در فایل JSON."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    timestamp: datetime = Field(default_factory=datetime.now)

class PendingTicketList(BaseModel):
    data: List[PendingTicket]

class AdminTicketResponsePayload(BaseModel):
    """مدلی که ادمین برای پاسخ به یک تیکت استفاده می‌کند."""
    question_id: str
    answer: str
class KnowledgeBaseItem(BaseModel):
    """مدل یک آیتم در پایگاه دانش برای نمایش به ادمین."""
    id: str
    question: str
    answer: str

class KnowledgeBaseDump(BaseModel):
    """مدل برای نمایش کل محتوای پایگاه دانش."""
    total_items: int
    data: List[KnowledgeBaseItem]