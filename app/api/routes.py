from fastapi import APIRouter, Depends
from app.services.models.agent import get_model, QAModel
from app.models.payload import QARequest
from app.models.prediction import QAResponse
router = APIRouter()
@router.post("/qna", response_model=QAResponse, name="qna")
def post_qna(
        request_payload: QARequest,
        model: QAModel = Depends(get_model)
) -> QAResponse:
    prediction_result = model.predict(request_payload.query)
    return QAResponse(**prediction_result)