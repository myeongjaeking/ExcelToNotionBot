from fastapi import HTTPException


class NotionException(HTTPException):
    """Notion 관련 예외"""
    def __init__(self, status_code: int = 500, detail: str = "Notion operation failed"):
        super().__init__(status_code=status_code, detail=detail)


class SlackException(HTTPException):
    """Slack 관련 예외"""
    def __init__(self, status_code: int = 500, detail: str = "Slack operation failed"):
        super().__init__(status_code=status_code, detail=detail)


class ExcelException(HTTPException):
    """Excel 파일 처리 관련 예외"""
    def __init__(self, status_code: int = 400, detail: str = "Excel file processing failed"):
        super().__init__(status_code=status_code, detail=detail)


class ConfigurationException(HTTPException):
    """설정 관련 예외"""
    def __init__(self, status_code: int = 500, detail: str = "Configuration error"):
        super().__init__(status_code=status_code, detail=detail)

