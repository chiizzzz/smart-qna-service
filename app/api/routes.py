# مسیر فایل: app/api/routes.py

from fastapi import APIRouter, Depends, status
from app.services.models.agent import get_model, QAModel
from app.models.payload import QARequest, QAList, DeleteRequest, OperatorAnswerPayload
from app.models.prediction import QAResponse

router = APIRouter()

# --- اندپوینت تسک ۱ و ۲ (بدون تغییر) ---
@router.post("/qna", response_model=QAResponse, name="qna")
def post_qna(request_payload: QARequest, model: QAModel = Depends(get_model)) -> QAResponse:
    prediction_result = model.predict(request_payload.query)
    return QAResponse(**prediction_result)

# === اندپوینت‌های جدید برای تسک ۳ و ۴ ===

# این اندپوینت از بخش "admin" است تا از اندپوینت عمومی جدا باشد
admin_router = APIRouter()

@admin_router.post(
    "/knowledge_base/add",
    status_code=status.HTTP_201_CREATED,
    name="kb:add_entries"
)
def add_entries(payload: QAList, model: QAModel = Depends(get_model)):
    """
    اندپوینت برای اضافه کردن لیستی از پرسش و پاسخ‌ها به پایگاه دانش.
    """
    result = model.add_entries(payload.dict()["data"])
    return result

@admin_router.put(
    "/knowledge_base/overwrite",
    status_code=status.HTTP_200_OK,
    name="kb:overwrite_database"
)
def overwrite_database(payload: QAList, model: QAModel = Depends(get_model)):
    """
    اندپوینت برای بازنویسی کامل پایگاه دانش با داده‌های جدید.
    """
    result = model.overwrite_database(payload.dict()["data"])
    return result

@admin_router.delete(
    "/knowledge_base/delete",
    status_code=status.HTTP_200_OK,
    name="kb:delete_entries"
)
def delete_entries(payload: DeleteRequest = None, model: QAModel = Depends(get_model)):
    """
    اندپوینت برای حذف آیتم‌ها.
    اگر بدنه درخواست خالی باشد، همه چیز را حذف می‌کند.
    اگر لیستی از ID ها داده شود، فقط آنها را حذف می‌کند.
    """
    ids = payload.ids if payload else None
    result = model.delete_entries(ids)
    return result

@admin_router.post(
    "/knowledge_base/add_from_operator",
    status_code=status.HTTP_201_CREATED,
    name="kb:add_from_operator"
)
def add_from_operator(payload: OperatorAnswerPayload, model: QAModel = Depends(get_model)):
    """
    اندپوینتی برای اینکه اپراتور پاسخ یک سوال بی‌پاسخ را به سیستم اضافه کند.
    """
    # ما از همان متد add_entries استفاده می‌کنیم و فقط یک آیتم به آن می‌دهیم
    result = model.add_entries([payload.dict()])
    return result

# برای ادغام روتر ادمین با روتر اصلی
router.include_router(admin_router, prefix="/admin", tags=["Knowledge Base Management"])