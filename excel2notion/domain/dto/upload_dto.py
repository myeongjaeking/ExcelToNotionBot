from typing import Optional, List, Dict
from pydantic import BaseModel


class UploadExcelRequest(BaseModel):
    """Excel 업로드 요청 DTO"""
    file_url: Optional[str] = None
    slack_token: Optional[str] = None
    database_id: Optional[str] = None


class UploadExcelResponse(BaseModel):
    """Excel 업로드 응답 DTO"""
    status: str
    message: str
    details: Dict


class UploadResult(BaseModel):
    """업로드 결과 DTO"""
    success: int
    failed: int
    errors: List[str]
    matched_columns: List[str]
    unmatched_columns: List[str]
    auto_created: List[str]
    auto_created_types: Dict[str, str]

