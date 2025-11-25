import logging
from typing import Tuple
import pandas as pd
from ..repository.slack_repository import SlackRepository
from ..utils.excel_parser import parse_excel_file

logger = logging.getLogger(__name__)


class ExcelService:
    """Excel 파일 처리 관련 비즈니스 로직을 담당하는 Service"""
    
    def __init__(self, slack_repository: SlackRepository):
        self.slack_repository = slack_repository
    
    def download_and_parse_file(self, file_id: str) -> Tuple[pd.DataFrame, str]:
        """Slack에서 파일 다운로드 후 Excel 파싱"""
        file_content, file_name = self.slack_repository.download_file(file_id)
        df = parse_excel_file(file_content)
        logger.info(f"Excel file parsed: {len(df)} rows, {len(df.columns)} columns")
        return df, file_name
    
    def download_and_parse_file_from_url(self, file_url: str, token: str) -> pd.DataFrame:
        """URL에서 파일 다운로드 후 Excel 파싱"""
        file_content = self.slack_repository.download_file_from_url(file_url, token)
        df = parse_excel_file(file_content)
        logger.info(f"Excel file parsed: {len(df)} rows, {len(df.columns)} columns")
        return df

