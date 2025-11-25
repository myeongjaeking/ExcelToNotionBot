import logging
import os
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
import faiss

logger = logging.getLogger(__name__)


class FAISSIndexManager:
    """FAISS 인덱스 관리 클래스"""
    
    def __init__(self, index_file_path: str = "faiss_index.bin", metadata_file_path: str = "faiss_metadata.json"):
        self.index_file_path = index_file_path
        self.metadata_file_path = metadata_file_path
        self.index: Optional[faiss.Index] = None
        self.metadata: Dict[str, Dict] = {}  # {page_id: {data: {...}, embedding_index: int}}
        self.dimension: Optional[int] = None
    
    def initialize_index(self, dimension: int):
        """FAISS 인덱스 초기화"""
        if self.index is not None:
            logger.warning("Index already initialized, skipping...")
            return
        
        self.dimension = dimension
        # L2 거리 기반 인덱스 (코사인 유사도는 정규화된 벡터에서 L2 거리와 동일)
        self.index = faiss.IndexFlatL2(dimension)
        logger.info(f"Initialized FAISS index with dimension {dimension}")
    
    def load_index(self) -> bool:
        """로컬 파일에서 인덱스 로드"""
        try:
            if not os.path.exists(self.index_file_path):
                logger.info("FAISS index file not found, will create new index")
                return False
            
            if not os.path.exists(self.metadata_file_path):
                logger.warning("FAISS metadata file not found")
                return False
            
            # 인덱스 로드
            self.index = faiss.read_index(self.index_file_path)
            self.dimension = self.index.d
            
            # 메타데이터 로드
            with open(self.metadata_file_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            logger.info(f"Loaded FAISS index with {len(self.metadata)} entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
            return False
    
    def save_index(self):
        """인덱스를 로컬 파일에 저장"""
        try:
            if self.index is None:
                logger.warning("No index to save")
                return
            
            # 인덱스 저장
            faiss.write_index(self.index, self.index_file_path)
            
            # 메타데이터 저장
            with open(self.metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved FAISS index with {len(self.metadata)} entries")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {str(e)}")
            raise
    
    def add_embedding(self, page_id: str, embedding: List[float], page_data: Dict) -> bool:
        """임베딩을 인덱스에 추가 (정규화 포함)"""
        try:
            if self.index is None:
                if self.dimension is None:
                    self.dimension = len(embedding)
                self.initialize_index(self.dimension)
            
            # 벡터를 numpy 배열로 변환
            embedding_array = np.array([embedding], dtype=np.float32)
            
            # 정규화 (코사인 유사도 계산을 위해)
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm
            
            # 인덱스에 추가
            embedding_index = self.index.ntotal
            self.index.add(embedding_array)
            
            # 메타데이터 저장
            self.metadata[page_id] = {
                "data": page_data,
                "embedding_index": int(embedding_index)
            }
            
            logger.info(f"Added embedding for page {page_id} at index {embedding_index}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embedding: {str(e)}")
            return False
    
    def search(self, query_embedding: List[float], k: int = 1) -> List[Tuple[str, Dict, float]]:
        """인덱스에서 가장 유사한 k개 검색"""
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("Index is empty or not initialized")
                return []
            
            # 쿼리 벡터를 numpy 배열로 변환
            query_array = np.array([query_embedding], dtype=np.float32)
            
            # 정규화 (코사인 유사도 계산을 위해)
            norm = np.linalg.norm(query_array)
            if norm > 0:
                query_array = query_array / norm
            
            # 검색 (거리와 인덱스 반환)
            distances, indices = self.index.search(query_array, k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # FAISS가 -1을 반환하는 경우 (결과 없음)
                    continue
                
                # 인덱스로부터 page_id 찾기
                page_id = None
                for pid, meta in self.metadata.items():
                    if meta["embedding_index"] == int(idx):
                        page_id = pid
                        break
                
                if page_id:
                    # L2 거리를 코사인 유사도로 변환 (정규화된 벡터의 경우)
                    # 정규화된 벡터: ||a|| = ||b|| = 1
                    # L2 거리^2 = ||a - b||^2 = 2 - 2*cos(θ)
                    # 따라서 cos(θ) = 1 - (L2 거리^2 / 2)
                    distance_squared = float(distance)
                    similarity = 1.0 - (distance_squared / 2.0)
                    similarity = max(0.0, min(1.0, similarity))  # 0-1 범위로 제한
                    
                    page_data = self.metadata[page_id]["data"]
                    results.append((page_id, page_data, similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search in FAISS index: {str(e)}")
            return []
    
    def get_all_page_ids(self) -> List[str]:
        """인덱스에 있는 모든 page_id 반환"""
        return list(self.metadata.keys())
    
    def remove_page(self, page_id: str) -> bool:
        """페이지 제거 (인덱스 재구성 필요 - 현재는 단순히 메타데이터만 제거)"""
        if page_id in self.metadata:
            del self.metadata[page_id]
            logger.info(f"Removed page {page_id} from metadata")
            return True
        return False
    
    def rebuild_index(self, all_embeddings: List[Tuple[str, List[float], Dict]]):
        """인덱스 재구성 (모든 임베딩으로부터)"""
        try:
            if not all_embeddings:
                logger.warning("No embeddings to rebuild index")
                return
            
            # 차원 확인
            self.dimension = len(all_embeddings[0][1])
            
            # 새 인덱스 생성
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = {}
            
            # 모든 임베딩 추가 (정규화 포함)
            embeddings_list = []
            for page_id, embedding, page_data in all_embeddings:
                # 정규화
                embedding_array = np.array(embedding, dtype=np.float32)
                norm = np.linalg.norm(embedding_array)
                if norm > 0:
                    embedding_array = embedding_array / norm
                
                embeddings_list.append(embedding_array.tolist())
                embedding_index = len(embeddings_list) - 1
                self.metadata[page_id] = {
                    "data": page_data,
                    "embedding_index": embedding_index
                }
            
            # 배치로 추가
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            self.index.add(embeddings_array)
            
            logger.info(f"Rebuilt FAISS index with {len(all_embeddings)} entries")
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {str(e)}")
            raise

