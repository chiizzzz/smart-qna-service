from typing import Union
from fastapi import APIRouter, Depends, status, BackgroundTasks

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
    result = model.handle_user_feedback(payload.dict())
    return FeedbackResponse(**result)

admin_router = APIRouter()

@admin_router.post(
    "/knowledge_base/add",
    status_code=status.HTTP_201_CREATED,
    name="kb:add_entries",
    summary="افزودن دسته‌ای پرسش و پاسخ",
)
def add_entries(payload: QAList, model: QAModel = Depends(get_model)):
    return model.add_entries(payload.dict()["data"])


@admin_router.put(
    "/knowledge_base/overwrite",
    status_code=status.HTTP_200_OK,
    name="kb:overwrite_database",
    summary="بازنویسی کامل پایگاه دانش",
)
def overwrite_database(payload: QAList, model: QAModel = Depends(get_model)):
    return model.overwrite_database(payload.dict()["data"])


@admin_router.delete(
    "/knowledge_base/delete",
    status_code=status.HTTP_200_OK,
    name="kb:delete_entries",
    summary="حذف آیتم‌ها از پایگاه دانش",
)
def delete_entries(payload: DeleteRequest = None, model: QAModel = Depends(get_model)):
    ids = payload.ids if payload else None
    return model.delete_entries(ids)


@admin_router.post(
    "/knowledge_base/add_from_operator",
    status_code=status.HTTP_201_CREATED,
    name="kb:add_from_operator",
    summary="افزودن یک پرسش و پاسخ توسط اپراتور",
)
def add_from_operator(payload: OperatorAnswerPayload, model: QAModel = Depends(get_model)):
    return model.add_entries([payload.dict()])


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
    from app.services.tools.tools import _read_db
    tickets_dict = _read_db()
    return PendingTicketList(data=list(tickets_dict.values()))


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
    background_tasks.add_task(model.handle_admin_response, payload.dict())
    return {"message": "پاسخ شما ثبت شد و پردازش آن در پس‌زمینه انجام می‌شود."}


router.include_router(
    admin_router, prefix="/admin", tags=["Admin: Knowledge Base & Ticketing"]
)