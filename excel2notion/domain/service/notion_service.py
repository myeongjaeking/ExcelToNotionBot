import logging
from typing import Dict
import pandas as pd
from ..repository.notion_repository import NotionRepository
from ..utils.property_detector import smart_detect_property_type
from ..utils.property_converter import convert_to_notion_properties
from ..exception.exceptions import NotionException

logger = logging.getLogger(__name__)


class NotionService:
    """Notion 관련 비즈니스 로직을 담당하는 Service"""
    
    def __init__(self, repository: NotionRepository):
        self.repository = repository
    
    def auto_create_properties(self, database_id: str, df: pd.DataFrame) -> Dict:
        """엑셀 컬럼 분석 후 Notion DB 속성 자동 생성 (스마트 타입 추론)"""
        try:
            # 기존 DB 정보 가져오기
            db_info = self.repository.retrieve_database(database_id)
            existing_properties = db_info.get("properties", {})
            
            # Title 속성 존재 여부 확인
            has_title = any(prop.get("type") == "title" for prop in existing_properties.values())
            
            # 새로 추가할 속성 정의
            new_properties = {}
            
            for idx, col in enumerate(df.columns):
                if col in existing_properties:
                    logger.info(f"Property '{col}' already exists, skipping...")
                    continue
                
                # 샘플 데이터 추출 (처음 20개)
                sample_values = df[col].head(20).tolist()
                
                # 첫 번째 컬럼이고 Title이 없으면 Title로 설정
                if idx == 0 and not has_title:
                    new_properties[col] = {"title": {}}
                    has_title = True
                    logger.info(f"Setting '{col}' as Title property")
                else:
                    # 스마트 타입 추론
                    prop_type = smart_detect_property_type(col, sample_values)
                    new_properties[col] = prop_type
                    logger.info(f"Detected type for '{col}': {list(prop_type.keys())[0]}")
            
            if not new_properties:
                logger.info("No new properties to add")
                return {"status": "success", "added": [], "types": {}}
            
            # Notion DB 업데이트
            logger.info(f"Adding {len(new_properties)} properties: {list(new_properties.keys())}")
            self.repository.update_database(database_id, new_properties)
            
            return {
                "status": "success",
                "added": list(new_properties.keys()),
                "types": {k: list(v.keys())[0] for k, v in new_properties.items()}
            }
        
        except Exception as e:
            logger.error(f"Failed to auto-create properties: {str(e)}")
            raise NotionException(status_code=500, detail=f"속성 자동 생성 실패: {str(e)}")
    
    def upload_dataframe(self, df: pd.DataFrame, database_id: str, auto_create: bool = True) -> Dict:
        """DataFrame 데이터를 Notion 데이터베이스에 업로드 (속성 자동 생성 옵션)"""
        # 속성 자동 생성
        creation_result = {"added": [], "types": {}}
        if auto_create:
            logger.info("Auto-creating Notion properties from Excel columns...")
            creation_result = self.auto_create_properties(database_id, df)
            logger.info(f"Property creation result: {creation_result}")
        
        # Notion DB 속성 정보 가져오기
        logger.info("Fetching Notion database properties...")
        db_properties = self.repository.get_database_properties(database_id)
        logger.info(f"Found {len(db_properties)} properties: {list(db_properties.keys())}")
        
        # 엑셀 컬럼 중 매핑 가능한 것들 확인
        excel_cols = set(df.columns)
        notion_props = set(db_properties.keys())
        matched_cols = excel_cols & notion_props
        unmatched_cols = excel_cols - notion_props
        
        logger.info(f"Matched columns: {matched_cols}")
        if unmatched_cols:
            logger.warning(f"Unmatched columns (will be skipped): {unmatched_cols}")
        
        results = {
            "success": 0,
            "failed": 0,
            "errors": [],
            "matched_columns": list(matched_cols),
            "unmatched_columns": list(unmatched_cols),
            "auto_created": creation_result.get("added", []),
            "auto_created_types": creation_result.get("types", {})
        }
        
        for idx, row in df.iterrows():
            try:
                properties = convert_to_notion_properties(row, df, db_properties)
                
                if not properties:
                    results["failed"] += 1
                    error_msg = f"Row {idx + 1} failed: No valid properties to upload"
                    results["errors"].append(error_msg)
                    logger.warning(error_msg)
                    continue
                
                # Notion에 페이지 생성
                self.repository.create_page(database_id, properties)
                results["success"] += 1
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"Uploaded {idx + 1} rows...")
                    
            except Exception as e:
                results["failed"] += 1
                error_msg = f"Row {idx + 1} failed: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
        
        return results

