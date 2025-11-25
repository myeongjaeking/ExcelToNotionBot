import logging
from typing import Dict, Optional
from notion_client import Client
from ..exception.exceptions import NotionException

logger = logging.getLogger(__name__)


class NotionRepository:
    """Notion API 접근을 담당하는 Repository"""
    
    def __init__(self, client: Optional[Client]):
        if not client:
            raise NotionException(status_code=500, detail="Notion client not initialized")
        self.client = client
    
    def get_database_properties(self, database_id: str) -> Dict[str, str]:
        """Notion Database의 속성 정보 가져오기"""
        try:
            db_info = self.client.databases.retrieve(database_id=database_id)
            properties = db_info.get("properties", {})
            
            # 속성명 → 속성타입 매핑
            property_map = {}
            for prop_name, prop_info in properties.items():
                property_map[prop_name] = prop_info.get("type")
            
            return property_map
        except Exception as e:
            logger.error(f"Failed to retrieve database properties: {str(e)}")
            raise NotionException(status_code=500, detail=f"Notion DB 조회 실패: {str(e)}")
    
    def create_page(self, database_id: str, properties: Dict):
        """Notion에 페이지 생성"""
        try:
            self.client.pages.create(
                parent={"database_id": database_id},
                properties=properties
            )
        except Exception as e:
            logger.error(f"Failed to create page: {str(e)}")
            raise NotionException(status_code=500, detail=f"페이지 생성 실패: {str(e)}")
    
    def update_database(self, database_id: str, properties: Dict):
        """Notion Database 속성 업데이트"""
        try:
            self.client.databases.update(
                database_id=database_id,
                properties=properties
            )
        except Exception as e:
            logger.error(f"Failed to update database: {str(e)}")
            raise NotionException(status_code=500, detail=f"데이터베이스 업데이트 실패: {str(e)}")
    
    def retrieve_database(self, database_id: str) -> Dict:
        """Notion Database 정보 조회"""
        try:
            return self.client.databases.retrieve(database_id=database_id)
        except Exception as e:
            logger.error(f"Failed to retrieve database: {str(e)}")
            raise NotionException(status_code=500, detail=f"데이터베이스 조회 실패: {str(e)}")

