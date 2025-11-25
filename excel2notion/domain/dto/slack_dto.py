from typing import Optional
from pydantic import BaseModel


class SlackCommandResponse(BaseModel):
    """Slack 커맨드 응답 DTO"""
    response_type: str
    text: str


class SlackEventResponse(BaseModel):
    """Slack 이벤트 응답 DTO"""
    status: str
    message: Optional[str] = None
    challenge: Optional[str] = None

