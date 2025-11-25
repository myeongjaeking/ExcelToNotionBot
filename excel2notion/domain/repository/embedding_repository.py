import logging
import json
from typing import List, Tuple, Dict, Optional
from notion_client import Client
from ..exception.exceptions import NotionException

logger = logging.getLogger(__name__)


class EmbeddingRepository:
    """Notion 데이터베이스의 임베딩 관리 Repository"""
    
    def __init__(self, client: Client):
        if not client:
            raise NotionException(status_code=500, detail="Notion client not initialized")
        self.client = client
    
    def get_all_pages_with_embeddings(self, database_id: str, embedding_property_name: str = "임베딩") -> List[Tuple[str, Dict[str, any], List[float]]]:
        """데이터베이스의 모든 페이지와 임베딩 가져오기 (더 이상 사용하지 않음 - FAISS에서만 읽음)"""
        # 임베딩은 이제 FAISS 인덱스에만 저장되므로 Notion에서 읽지 않음
        # 이 메서드는 호환성을 위해 유지하지만 빈 리스트 반환
        logger.info("get_all_pages_with_embeddings: Embeddings are now stored in FAISS only, returning empty list")
        return []
    
    def get_all_pages(self, database_id: str) -> List[Tuple[str, Dict[str, any]]]:
        """데이터베이스의 모든 페이지 가져오기 (임베딩 없이)"""
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
                    page_id = page.get("id")
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
                    
                    pages.append((page_id, page_data))
                
                has_more = response.get("has_more", False)
                start_cursor = response.get("next_cursor")
            
            return pages
        except Exception as e:
            logger.error(f"Failed to get all pages: {str(e)}")
            raise NotionException(status_code=500, detail=f"페이지 조회 실패: {str(e)}")
    
    def save_embedding_to_page(
        self, 
        page_id: str, 
        embedding: List[float], 
        embedding_property_name: str = "임베딩"
    ):
        """페이지에 임베딩 저장 (더 이상 사용하지 않음 - FAISS에만 저장)"""
        # Notion의 rich_text 필드는 최대 2000자까지만 저장 가능하므로
        # 임베딩은 FAISS 인덱스에만 저장하고 Notion에는 저장하지 않음
        logger.info(f"Skipping embedding save to Notion for page {page_id} (stored in FAISS only)")
        pass
    
    def create_page_with_embedding(
        self,
        database_id: str,
        properties: Dict,
        embedding: List[float],
        embedding_property_name: str = "임베딩"
    ) -> str:
        """임베딩과 함께 페이지 생성 (임베딩은 FAISS에만 저장)"""
        try:
            # Notion의 rich_text 필드는 최대 2000자까지만 저장 가능하므로
            # 임베딩은 FAISS 인덱스에만 저장하고 Notion에는 저장하지 않음
            # properties에서 임베딩 속성 제거 (있다면)
            if embedding_property_name in properties:
                del properties[embedding_property_name]
            
            # 페이지 생성
            response = self.client.pages.create(
                parent={"database_id": database_id},
                properties=properties
            )
            
            return response.get("id")
        except Exception as e:
            logger.error(f"Failed to create page: {str(e)}")
            raise NotionException(status_code=500, detail=f"페이지 생성 실패: {str(e)}")

