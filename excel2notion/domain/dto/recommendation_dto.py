from typing import Optional, Dict, Any
from pydantic import BaseModel


class Append2Top1Request(BaseModel):
    """append2top1 요청 DTO"""
    text: Optional[str] = None
    file_content: Optional[bytes] = None


class Append2Top1Response(BaseModel):
    """append2top1 응답 DTO"""
    success: bool
    similarity_score: float
    top1_restaurant: str
    recommendation_reason: str
    new_page_id: str
    structured_data: Dict[str, Any]
    top1_data: Dict[str, Any]

