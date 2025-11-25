from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check 응답 DTO"""
    status: str
    notion_configured: bool
    slack_configured: bool

