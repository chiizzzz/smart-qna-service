from pydantic import BaseModel
class QARequest(BaseModel):
    query: str