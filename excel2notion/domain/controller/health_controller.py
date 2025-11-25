from fastapi import APIRouter
from fastapi.responses import JSONResponse
from ..config.settings import get_notion_client, get_slack_client

router = APIRouter(tags=["health"])


@router.post("/health")
async def health_check():
    """Health check 엔드포인트"""
    return JSONResponse(content={
        "status": "healthy",
        "notion_configured": get_notion_client() is not None,
        "slack_configured": get_slack_client() is not None
    })

