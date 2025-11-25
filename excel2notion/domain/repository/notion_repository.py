import logging
from typing import Dict, Optional,List
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
    
    def get_all_pages_data(self, database_id: str) -> List[Dict]:
        """데이터베이스의 모든 페이지 데이터 가져오기"""
        try:
            pages = []
            has_more = True
            start_cursor = None
            
            while has_more:
                if start_cursor:
                    response = self.client.databases.query(
                        database_id=database_id,
                        start_cursor=start_cursor
                    )
                else:
                    response = self.client.databases.query(database_id=database_id)
                
                for page in response.get("results", []):
                    properties = page.get("properties", {})
                    
                    # 페이지 데이터 추출
                    page_data = {}
                    for prop_name, prop_info in properties.items():
                        prop_type = prop_info.get("type")
                        if prop_type == "title":
                            title = prop_info.get("title", [])
                            page_data[prop_name] = title[0].get("plain_text", "") if title else ""
                        elif prop_type == "rich_text":
                            rich_text = prop_info.get("rich_text", [])
                            page_data[prop_name] = rich_text[0].get("plain_text", "") if rich_text else ""
                        elif prop_type == "number":
                            page_data[prop_name] = prop_info.get("number")
                        elif prop_type == "select":
                            select = prop_info.get("select")
                            page_data[prop_name] = select.get("name", "") if select else ""
                        elif prop_type == "multi_select":
                            multi_select = prop_info.get("multi_select", [])
                            page_data[prop_name] = ", ".join([item.get("name", "") for item in multi_select])
                        elif prop_type == "date":
                            date = prop_info.get("date")
                            if date:
                                page_data[prop_name] = date.get("start", "")
                        elif prop_type == "checkbox":
                            page_data[prop_name] = prop_info.get("checkbox", False)
                        elif prop_type == "url":
                            page_data[prop_name] = prop_info.get("url", "")
                        elif prop_type == "email":
                            page_data[prop_name] = prop_info.get("email", "")
                        elif prop_type == "phone_number":
                            page_data[prop_name] = prop_info.get("phone_number", "")
                    
                    pages.append(page_data)
                
                has_more = response.get("has_more", False)
                start_cursor = response.get("next_cursor")
            
            return pages
        except Exception as e:
            logger.error(f"Failed to get all pages data: {str(e)}")
            raise NotionException(status_code=500, detail=f"페이지 데이터 조회 실패: {str(e)}")

