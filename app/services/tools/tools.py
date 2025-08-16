# In: app/services/tools/tools.py
import json
import threading
from app.schemas import PendingTicket
from typing import List, Optional, Dict, Literal
file_lock = threading.Lock()
PENDING_TICKETS_DB = "pending_tickets.json"


def _read_db() -> Dict[str, Dict]:
    """پایگاه داده تیکت‌ها را از فایل می‌خواند و به صورت دیکشنری برمی‌گرداند."""
    try:
        with open(PENDING_TICKETS_DB, "r", encoding="utf-8") as f:
            # { "data": [ ... ] }
            tickets_list = json.load(f).get("data", [])
            # تبدیل لیست به دیکشنری برای دسترسی سریع با ID
            return {ticket["question_id"]: ticket for ticket in tickets_list}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_db(tickets_dict: Dict[str, Dict]):
    """یک دیکشنری از تیکت‌ها را به فایل JSON می‌نویسد."""
    with open(PENDING_TICKETS_DB, "w", encoding="utf-8") as f:
        # تبدیل دوباره به فرمت لیست برای ذخیره‌سازی
        tickets_list = list(tickets_dict.values())
        json.dump({"data": tickets_list}, f, ensure_ascii=False, indent=2, default=str)


def create_ticket(
    question: str,
    bot_answer: Optional[str] = None,
    source: Literal["unanswered", "negative_feedback"] = "unanswered"
) -> PendingTicket:
    """یک تیکت جدید برای یک سوال بی‌پاسخ یا با پاسخ اشتباه ایجاد و ذخیره می‌کند."""
    print(f"INFO: در حال ایجاد تیکت برای سوال: '{question}' (منبع: {source})")

    new_ticket = PendingTicket(
        question=question,
        bot_answer=bot_answer,
        ticket_source=source
    )

    with file_lock:
        db = _read_db()
        db[new_ticket.question_id] = new_ticket.dict()
        _write_db(db)

    print(f"INFO: تیکت با ID {new_ticket.question_id} با موفقیت ایجاد شد.")
    return new_ticket



def close_ticket(question_id: str):
    """یک تیکت را پس از پاسخ‌دهی از پایگاه داده حذف می‌کند."""
    with file_lock:
        db = _read_db()
        if question_id in db:
            del db[question_id]
            _write_db(db)
            print(f"INFO: تیکت {question_id} بسته و از لیست انتظار حذف شد.")