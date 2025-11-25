import logging
import requests
from typing import Tuple, Optional, List, Dict
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from ..exception.exceptions import SlackException
from ..config.settings import get_slack_bot_token

logger = logging.getLogger(__name__)


class SlackRepository:
    """Slack API 접근을 담당하는 Repository"""
    
    def __init__(self, client: Optional[WebClient]):
        if not client:
            raise SlackException(status_code=500, detail="Slack client not initialized")
        self.client = client
    
    def download_file(self, file_id: str) -> Tuple[bytes, str]:
        """Slack 파일 ID로 파일 다운로드"""
        try:
            # 파일 정보 가져오기
            file_info = self.client.files_info(file=file_id)
            file_data = file_info["file"]
            file_name = file_data.get("name", "file.xlsx")
            file_url_private = file_data.get("url_private_download")
            
            if not file_url_private:
                raise SlackException(status_code=400, detail="File download URL not available")
            
            # requests로 직접 다운로드
            token = get_slack_bot_token()
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(file_url_private, headers=headers)
            response.raise_for_status()
            file_content = response.content
            
            return file_content, file_name
        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            raise SlackException(status_code=500, detail=f"Slack API error: {e.response['error']}")
        except Exception as e:
            logger.error(f"Failed to download file: {str(e)}")
            raise SlackException(status_code=500, detail=f"파일 다운로드 실패: {str(e)}")
    
    def download_file_from_url(self, file_url: str, token: str) -> bytes:
        """Slack 파일 URL에서 파일 다운로드 (대체 방법)"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(file_url, headers=headers)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to download file from URL: {str(e)}")
            raise SlackException(status_code=500, detail=f"파일 다운로드 실패: {str(e)}")
    
    def list_files(self, channel_id: str, file_types: str = "xlsx", count: int = 10) -> List[Dict]:
        """채널의 파일 목록 가져오기"""
        try:
            files_response = self.client.files_list(
                channel=channel_id,
                types=file_types,
                count=count
            )
            return files_response.get("files", [])
        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            raise SlackException(status_code=500, detail=f"Slack API error: {e.response['error']}")
    
    def post_message(self, channel_id: str, text: str):
        """채널에 메시지 전송"""
        try:
            self.client.chat_postMessage(
                channel=channel_id,
                text=text
            )
        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            raise SlackException(status_code=500, detail=f"Slack API error: {e.response['error']}")

