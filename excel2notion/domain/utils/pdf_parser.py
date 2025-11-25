import logging
import tempfile
import os
from typing import Optional
import PyPDF2
from io import BytesIO

logger = logging.getLogger(__name__)


def is_pdf(content: bytes) -> bool:
    """파일 내용이 PDF인지 확인"""
    return content[:4] == b'%PDF'


def parse_pdf(file_content: bytes) -> str:
    """PDF 파일을 텍스트로 변환"""
    try:
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_content = []
        for page in pdf_reader.pages:
            text_content.append(page.extract_text())
        
        return "\n".join(text_content)
    except Exception as e:
        logger.error(f"Failed to parse PDF: {str(e)}")
        raise ValueError(f"PDF 파싱 실패: {str(e)}")


def detect_input_type(content: Optional[bytes], text: Optional[str]) -> str:
    """입력 타입 감지 (자연어 또는 PDF)"""
    if content and is_pdf(content):
        return "pdf"
    elif text:
        return "natural_language"
    else:
        raise ValueError("입력값이 없습니다.")


def extract_text_from_input(content: Optional[bytes], text: Optional[str]) -> str:
    """입력에서 텍스트 추출"""
    input_type = detect_input_type(content, text)
    
    if input_type == "pdf":
        if not content:
            raise ValueError("PDF 파일 내용이 필요합니다.")
        return parse_pdf(content)
    else:
        if not text:
            raise ValueError("자연어 텍스트가 필요합니다.")
        return text

