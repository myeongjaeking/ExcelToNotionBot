import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def convert_to_notion_properties(row: pd.Series, df: pd.DataFrame, db_properties: Dict) -> Dict:
    """DataFrame의 행을 Notion 속성 형식으로 변환 (자동 매핑)"""
    properties = {}
    
    for col in df.columns:
        value = row[col]
        
        # NaN 값 스킵
        if pd.isna(value):
            continue
        
        # Notion DB에 해당 컬럼명이 존재하는지 확인
        if col not in db_properties:
            logger.warning(f"Column '{col}' not found in Notion DB properties, skipping...")
            continue
        
        prop_type = db_properties[col]
        
        # 속성 타입별로 변환
        if prop_type == "title":
            properties[col] = {
                "title": [
                    {
                        "text": {
                            "content": str(value)[:2000]  # Notion title 길이 제한
                        }
                    }
                ]
            }
        elif prop_type == "rich_text":
            properties[col] = {
                "rich_text": [
                    {
                        "text": {
                            "content": str(value)[:2000]  # Notion text 길이 제한
                        }
                    }
                ]
            }
        elif prop_type == "number":
            try:
                properties[col] = {
                    "number": float(value)
                }
            except (ValueError, TypeError):
                logger.warning(f"Cannot convert '{value}' to number for column '{col}'")
                continue
        elif prop_type == "checkbox":
            properties[col] = {
                "checkbox": bool(value)
            }
        elif prop_type == "date":
            try:
                if pd.notna(value):
                    date_str = pd.Timestamp(value).strftime("%Y-%m-%d")
                    properties[col] = {
                        "date": {
                            "start": date_str
                        }
                    }
            except Exception as e:
                logger.warning(f"Cannot convert '{value}' to date for column '{col}': {e}")
                continue
        elif prop_type == "select":
            properties[col] = {
                "select": {
                    "name": str(value)[:100]  # Select option 길이 제한
                }
            }
        elif prop_type == "multi_select":
            # 쉼표로 구분된 값을 multi_select로 변환
            options = [opt.strip() for opt in str(value).split(",") if opt.strip()]
            properties[col] = {
                "multi_select": [
                    {"name": opt[:100]} for opt in options[:10]  # 최대 10개 옵션
                ]
            }
        elif prop_type == "url":
            properties[col] = {
                "url": str(value)[:2000]
            }
        elif prop_type == "email":
            properties[col] = {
                "email": str(value)[:200]
            }
        elif prop_type == "phone_number":
            properties[col] = {
                "phone_number": str(value)[:50]
            }
        else:
            # 지원하지 않는 타입은 rich_text로 변환
            logger.warning(f"Unsupported property type '{prop_type}' for column '{col}', converting to rich_text")
            properties[col] = {
                "rich_text": [
                    {
                        "text": {
                            "content": str(value)[:2000]
                        }
                    }
                ]
            }
    
    return properties

