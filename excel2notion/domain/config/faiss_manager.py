import logging
from ..utils.faiss_index import FAISSIndexManager
from ..config.settings import get_notion_database_id

logger = logging.getLogger(__name__)

# 전역 FAISS 인덱스 매니저 인스턴스
_faiss_manager: FAISSIndexManager = None


def get_faiss_manager() -> FAISSIndexManager:
    """FAISS 인덱스 매니저 싱글톤 인스턴스 반환"""
    global _faiss_manager
    if _faiss_manager is None:
        _faiss_manager = FAISSIndexManager()
    return _faiss_manager


def initialize_faiss_index(database_id: str = None):
    """서버 시작 시 FAISS 인덱스 초기화 및 로드"""
    global _faiss_manager
    
    try:
        manager = get_faiss_manager()
        
        # 로컬 파일에서 인덱스 로드 시도
        if manager.load_index():
            logger.info("FAISS index loaded successfully from file")
            return
        
        # 인덱스 파일이 없으면 Notion에서 데이터를 가져와서 인덱스 생성
        logger.info("FAISS index file not found, will be created on first use")
        
    except Exception as e:
        logger.error(f"Failed to initialize FAISS index: {str(e)}")

