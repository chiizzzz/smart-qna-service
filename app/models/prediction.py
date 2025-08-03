from pydantic import BaseModel
from typing import List
class QAResponse(BaseModel):
    tags_identified: List[str]
    final_answer: str
    retrieved_context_count: int