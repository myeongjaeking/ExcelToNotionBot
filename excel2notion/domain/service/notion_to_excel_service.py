import logging
from typing import List, Dict
from ..repository.notion_repository import NotionRepository
from ..utils.excel_writer import create_excel_from_data

logger = logging.getLogger(__name__)


class NotionToExcelService:
    """Notion 데이터를 Excel로 변환하는 Service"""
    
    def __init__(self, notion_repo: NotionRepository):
        self.notion_repo = notion_repo
    
    def export_to_excel(self, database_id: str) -> bytes:
        """Notion 데이터베이스를 Excel 파일로 변환"""
        try:
            logger.info(f"Fetching all pages from Notion database: {database_id}")
            # 모든 페이지 데이터 가져오기
            pages_data = self.notion_repo.get_all_pages_data(database_id)
            
            if not pages_data:
                raise ValueError("Notion 데이터베이스에 데이터가 없습니다.")
            
            logger.info(f"Found {len(pages_data)} pages. Converting to Excel...")
            # Excel 파일 생성
            excel_content = create_excel_from_data(pages_data)
            
            logger.info(f"Excel file created successfully ({len(excel_content)} bytes)")
            return excel_content
            
        except Exception as e:
            logger.error(f"Failed to export Notion to Excel: {str(e)}")
            raise

