import logging
from typing import Optional
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse

from ..service.excel_service import ExcelService
from ..service.notion_service import NotionService
from ..repository.slack_repository import SlackRepository
from ..config.settings import get_notion_database_id, get_slack_client, get_notion_client
from ..exception.exceptions import ConfigurationException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])


@router.post("/upload-excel")
async def upload_excel(
    file_url: Optional[str] = Form(None),
    slack_token: Optional[str] = Form(None),
    database_id: Optional[str] = Form(None)
):
    """Excel 파일을 업로드하고 Notion에 전송"""
    try:
        if not file_url:
            raise HTTPException(status_code=400, detail="file_url is required")
        
        if not slack_token:
            raise HTTPException(status_code=400, detail="slack_token is required")
        
        db_id = database_id or get_notion_database_id()
        if not db_id:
            raise HTTPException(status_code=400, detail="database_id is required")
        
        # 파일 다운로드
        logger.info(f"Downloading file from: {file_url}")
        slack_client = get_slack_client()
        if not slack_client:
            raise ConfigurationException(detail="Slack 클라이언트가 초기화되지 않았습니다.")
        
        slack_repo = SlackRepository(slack_client)
        excel_service = ExcelService(slack_repo)
        
        df = excel_service.download_and_parse_file_from_url(file_url, slack_token)
        
        # Notion에 업로드 (속성 자동 생성)
        logger.info("Uploading to Notion...")
        notion_client = get_notion_client()
        if not notion_client:
            raise ConfigurationException(detail="Notion 클라이언트가 초기화되지 않았습니다.")
        
        from ..repository.notion_repository import NotionRepository
        notion_repo = NotionRepository(notion_client)
        notion_service = NotionService(notion_repo)
        results = notion_service.upload_dataframe(df, db_id, auto_create=True)
        
        message = f"업로드 완료: 성공 {results['success']}개, 실패 {results['failed']}개"
        if results['auto_created']:
            message += f" | 자동 생성된 속성: {', '.join(results['auto_created'])}"
        
        return JSONResponse(content={
            "status": "success",
            "message": message,
            "details": results
        })
        
    except Exception as e:
        logger.error(f"Error uploading Excel: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

