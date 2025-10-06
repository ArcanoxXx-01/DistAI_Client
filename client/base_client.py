from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseClient(ABC):
    """Base interface for DistIA clients"""

    @abstractmethod
    def upload_dataset(self, file_path: str, name: str = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def create_job(
        self,
        dataset_id: str,
        task: str,
        models: list = [],
        params: Optional[Dict[str, Any]] = None,
        train_test_split: float = 0.2,
        seed: int = 42,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def list_jobs(self, user_id: str = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def download_model(self, job_id: str, output_path: str = None) -> str:
        pass
