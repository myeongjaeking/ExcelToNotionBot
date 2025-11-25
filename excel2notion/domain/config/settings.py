import os
import logging
from dotenv import load_dotenv
from notion_client import Client
from slack_sdk import WebClient
from typing import Optional

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경변수에서 설정 가져오기
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def get_notion_client() -> Optional[Client]:
    """Notion 클라이언트 반환"""
    if NOTION_TOKEN:
        return Client(auth=NOTION_TOKEN)
    return None


def get_slack_client() -> Optional[WebClient]:
    """Slack 클라이언트 반환"""
    if SLACK_BOT_TOKEN:
        return WebClient(token=SLACK_BOT_TOKEN)
    return None


def get_notion_database_id() -> Optional[str]:
    """Notion 데이터베이스 ID 반환"""
    return NOTION_DATABASE_ID


def get_slack_bot_token() -> Optional[str]:
    """Slack Bot Token 반환"""
    return SLACK_BOT_TOKEN


def get_gemini_api_key() -> Optional[str]:
    """Gemini API Key 반환"""
    return GEMINI_API_KEY

