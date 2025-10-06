import requests, os, json
from typing import Dict, Any, Optional
from client.base_client import BaseClient
from client.centralized.config_manager import load_config, save_config

DEFAULT_TIMEOUT = 10


class HttpClient(BaseClient):
    def __init__(self):
        self.cfg = load_config()
        self.server = self._discover_server()
        self.token = self.cfg.get("token")
        if not self.token:
            self.token = self._request_token()
            if self.token:
                self.cfg["token"] = self.token
                save_config(self.cfg)

    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _discover_server(self) -> str:
        servers = self.cfg.get("servers", [])
        for s in servers:
            try:
                r = requests.get(f"{s}/api/ping", timeout=DEFAULT_TIMEOUT)
                if r.status_code == 200:
                    print(f"Conectado a servidor: {s}")
                    return s
            except Exception:
                continue
        raise RuntimeError(
            "No hay servidores disponibles. Revisa ~/client/centralized/config.json"
        )

    def _request_token(self) -> Optional[str]:
        # simple request; server accepts JSON or form
        username = os.getenv("CLIENT_USER", "user1")
        try:
            r = requests.post(
                f"{self.server}/api/token",
                json={"username": username},
                timeout=DEFAULT_TIMEOUT,
            )
            r.raise_for_status()
            return r.json().get("token")
        except Exception:
            # fallback to form
            try:
                r = requests.post(
                    f"{self.server}/api/token",
                    data={"username": username},
                    timeout=DEFAULT_TIMEOUT,
                )
                r.raise_for_status()
                return r.json().get("token")
            except Exception as e:
                print("No se pudo obtener token:", e)
                return None

    def upload_dataset(self, file_path: str, name: str = None) -> Dict[str, Any]:
        url = f"{self.server}/api/datasets/upload"
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        files = {
            "file": (os.path.basename(file_path), open(file_path, "rb"), "text/csv")
        }
        data = {"name": name or os.path.basename(file_path)}
        r = requests.post(
            url,
            headers=self._headers(),
            files=files,
            data=data,
            timeout=DEFAULT_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    def create_job(
        self,
        dataset_id: str,
        task: str,
        model: str,
        params: Dict[str, Any] = None,
        train_test_split: float = 0.2,
        seed: int = 42,
    ) -> Dict[str, Any]:
        url = f"{self.server}/api/jobs"
        payload = {
            "dataset_id": dataset_id,
            "task": task,
            "model": model,
            "params": params or {},
            "train_test_split": train_test_split,
            "seed": seed,
        }
        r = requests.post(
            url,
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        url = f"{self.server}/api/jobs/{job_id}"
        r = requests.get(url, headers=self._headers(), timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        return r.json()

    def list_jobs(self, user_id: str = None) -> Dict[str, Any]:
        url = f"{self.server}/api/jobs"
        params = {"user_id": user_id} if user_id else {}
        r = requests.get(
            url, headers=self._headers(), params=params, timeout=DEFAULT_TIMEOUT
        )
        r.raise_for_status()
        return r.json()

    def download_model(self, job_id: str, output_path: str = None) -> str:
        url = f"{self.server}/api/jobs/{job_id}/model"
        r = requests.get(
            url, headers=self._headers(), stream=True, timeout=DEFAULT_TIMEOUT
        )
        r.raise_for_status()
        out_path = output_path or f"model_{job_id}.pkl"
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return out_path

    def update_server_list(self):
        url = f"{self.server}/api/cluster/nodes"
        r = requests.get(url, headers=self._headers(), timeout=DEFAULT_TIMEOUT)
        if r.status_code == 200:
            nodes = r.json().get("nodes", [])
            self.cfg["servers"] = nodes
            save_config(self.cfg)
            print("Lista de servidores actualizada:", nodes)
