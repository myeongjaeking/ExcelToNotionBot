import pandas as pd
from typing import List, Dict


def smart_detect_property_type(column_name: str, sample_values: List) -> Dict:
    """컬럼명과 샘플 데이터로 속성 타입 자동 추론"""
    column_lower = column_name.lower()
    
    # 컬럼명 기반 타입 추론
    if any(keyword in column_lower for keyword in ["가격", "금액", "price", "cost", "amount"]):
        return {"number": {"format": "won"}}
    
    if any(keyword in column_lower for keyword in ["평점", "점수", "rating", "score"]):
        return {"number": {"format": "number"}}
    
    if any(keyword in column_lower for keyword in ["날짜", "date", "day"]):
        return {"date": {}}
    
    if any(keyword in column_lower for keyword in ["url", "링크", "link"]):
        return {"url": {}}
    
    if any(keyword in column_lower for keyword in ["이메일", "email", "mail"]):
        return {"email": {}}
    
    if any(keyword in column_lower for keyword in ["전화", "phone", "tel"]):
        return {"phone_number": {}}
    
    # 샘플 데이터 기반 타입 추론
    non_null_values = [v for v in sample_values if pd.notna(v)]
    
    if not non_null_values:
        return {"rich_text": {}}
    
    # 모두 숫자인지 확인
    try:
        all_numbers = all(isinstance(v, (int, float)) or str(v).replace('.', '', 1).replace('-', '', 1).isdigit() 
                         for v in non_null_values[:10])
        if all_numbers:
            return {"number": {}}
    except:
        pass
    
    # 모두 True/False인지 확인
    try:
        all_bools = all(str(v).lower() in ['true', 'false', '0', '1', 'yes', 'no'] 
                       for v in non_null_values[:10])
        if all_bools:
            return {"checkbox": {}}
    except:
        pass
    
    # 기본값: rich_text
    return {"rich_text": {}}

