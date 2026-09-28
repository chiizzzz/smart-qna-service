import secrets
from typing import Optional, Union
from fastapi import APIRouter, Depends, status, BackgroundTasks, Header, HTTPException

from app.core.config import settings
from app.services.tools import tools

from app.services.models.agent import get_model, QAModel
from app.schemas import (
    QARequest,
    QAResponse,
    QAList,
    DeleteRequest,
    OperatorAnswerPayload,
    TicketCreationResponse,
    PendingTicketList,
    AdminTicketResponsePayload,
    KnowledgeBaseDump,
    KnowledgeBaseItem,
    FeedbackRequest,
    FeedbackResponse,
)

router = APIRouter()

@router.post(
    "/qna",
    response_model=Union[QAResponse, TicketCreationResponse],
    name="qna",
    summary="دریافت پرسش و پاسخ اصلی",
)
def post_qna(
    request_payload: QARequest, model: QAModel = Depends(get_model)
) -> Union[QAResponse, TicketCreationResponse]:
    prediction_result = model.predict(request_payload.query)

    if prediction_result.get("status") == "ticket_created":
        return TicketCreationResponse(**prediction_result)

    return QAResponse(**prediction_result)

@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    name="feedback",
    summary="ثبت بازخورد کاربر در مورد پاسخ",
)
def post_feedback(
    payload: FeedbackRequest, model: QAModel = Depends(get_model)
) -> FeedbackResponse:
    result = model.handle_user_feedback(payload.model_dump())
    return FeedbackResponse(**result)

def require_admin(x_admin_token: Optional[str] = Header(default=None)):
    """مسیرهای ادمین فقط با هدر X-Admin-Token معتبر قابل دسترسی هستند."""
    if not settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ADMIN_API_KEY تنظیم نشده است؛ مسیرهای ادمین غیرفعال هستند.",
        )
    if not x_admin_token or not secrets.compare_digest(x_admin_token, settings.ADMIN_API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="توکن ادمین نامعتبر است.")


admin_router = APIRouter(dependencies=[Depends(require_admin)])

@admin_router.post(
    "/knowledge_base/add",
    status_code=status.HTTP_201_CREATED,
    name="kb:add_entries",
    summary="افزودن دسته‌ای پرسش و پاسخ",
)
def add_entries(payload: QAList, model: QAModel = Depends(get_model)):
    return model.add_entries(payload.model_dump()["data"])


@admin_router.put(
    "/knowledge_base/overwrite",
    status_code=status.HTTP_200_OK,
    name="kb:overwrite_database",
    summary="بازنویسی کامل پایگاه دانش",
)
def overwrite_database(payload: QAList, model: QAModel = Depends(get_model)):
    return model.overwrite_database(payload.model_dump()["data"])


@admin_router.delete(
    "/knowledge_base/delete",
    status_code=status.HTTP_200_OK,
    name="kb:delete_entries",
    summary="حذف آیتم‌ها از پایگاه دانش",
)
def delete_entries(payload: Optional[DeleteRequest] = None, model: QAModel = Depends(get_model)):
    ids = payload.ids if payload else None
    return model.delete_entries(ids)


@admin_router.post(
    "/knowledge_base/add_from_operator",
    status_code=status.HTTP_201_CREATED,
    name="kb:add_from_operator",
    summary="افزودن یک پرسش و پاسخ توسط اپراتور",
)
def add_from_operator(payload: OperatorAnswerPayload, model: QAModel = Depends(get_model)):
    return model.add_entries([payload.model_dump()])


@admin_router.get(
    "/knowledge_base/all",
    response_model=KnowledgeBaseDump,
    name="kb:get_all_entries",
    summary="دریافت تمام آیتم‌های موجود در پایگاه دانش",
)
def get_all_entries(model: QAModel = Depends(get_model)):
    all_items = model.get_all_entries()
    return KnowledgeBaseDump(
        total_items=len(all_items),
        data=[KnowledgeBaseItem(**item) for item in all_items]
    )

@admin_router.get(
    "/pending_tickets",
    response_model=PendingTicketList,
    name="admin:get_pending_tickets",
    summary="مشاهده لیست تیکت‌های در انتظار پاسخ",
)
def get_pending_tickets():
    return PendingTicketList(data=tools.list_tickets())


@admin_router.post(
    "/respond_to_ticket",
    status_code=status.HTTP_202_ACCEPTED,
    name="admin:respond_to_ticket",
    summary="پاسخ به یک تیکت مشخص",
)
def respond_to_ticket(
    payload: AdminTicketResponsePayload,
    background_tasks: BackgroundTasks,
    model: QAModel = Depends(get_model),
):
    if not tools.get_ticket(payload.question_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="تیکتی با این شناسه یافت نشد.")
    background_tasks.add_task(model.handle_admin_response, payload.model_dump())
    return {"message": "پاسخ شما ثبت شد و پردازش آن در پس‌زمینه انجام می‌شود."}


router.include_router(
    admin_router, prefix="/admin", tags=["Admin: Knowledge Base & Ticketing"]
)
