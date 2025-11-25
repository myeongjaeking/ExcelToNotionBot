import os
import tempfile
import pandas as pd
from typing import Tuple


def parse_excel_file(file_content: bytes) -> pd.DataFrame:
    """Excel 파일을 pandas DataFrame으로 변환"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name
    
    try:
        df = pd.read_excel(tmp_file_path, engine='openpyxl')
        return df
    finally:
        os.unlink(tmp_file_path)

