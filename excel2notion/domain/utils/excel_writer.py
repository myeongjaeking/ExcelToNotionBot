import logging
import pandas as pd
from typing import List, Dict
from io import BytesIO

logger = logging.getLogger(__name__)


def create_excel_from_data(data: List[Dict]) -> bytes:
    """데이터를 Excel 파일로 변환"""
    try:
        if not data:
            raise ValueError("데이터가 없습니다.")
        
        # DataFrame 생성
        df = pd.DataFrame(data)
        
        # Excel 파일을 메모리에 생성
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        
        output.seek(0)
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Failed to create Excel file: {str(e)}")
        raise ValueError(f"Excel 파일 생성 실패: {str(e)}")

