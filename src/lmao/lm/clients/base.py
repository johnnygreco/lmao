from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

__all__ = ["Client", "SUCCESS_STATUS_CODE"]
SUCCESS_STATUS_CODE = 200


class LM(ABC):
    @abstractmethod
    def chat(self, prompt: str, **kwargs) -> str:
        ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        ...


class Client(LM, ABC):
    base_url: str = "none"

    #  If the backoff_factor is 0.1, then sleep() will sleep for [0.0s, 0.2s, 0.4s, …] between retries.
    RETRY_BACKOFF_FACTOR: float = 0.1
    RETRY_STATUS_CODES: List[int] = [429, 500, 502, 503, 504]

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 5):
        self.max_retries = max_retries
        if api_key is None:
            raise ValueError("You must provide an API key to initialize an LM Client.")
        self.__api_key = api_key
        if self.base_url == "none":
            raise ValueError("All Client subclasses must define a base URL attribute.")

    def _post_request(self, api_path: str, request: dict, **extra_header_kwargs) -> Tuple[int, dict]:
        with requests.Session() as session:
            retries = Retry(
                total=self.max_retries,
                backoff_factor=self.RETRY_BACKOFF_FACTOR,
                status_forcelist=self.RETRY_STATUS_CODES,
            )
            session.mount("https://", HTTPAdapter(max_retries=retries))
            session.mount("http://", HTTPAdapter(max_retries=retries))
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {self.__api_key}",
            }
            headers.update(extra_header_kwargs)
        response = requests.post(url=f"{self.base_url}/{api_path}", json=request, headers=headers)
        status_code = response.status_code
        try:
            response_dict = response.json()
        except requests.exceptions.JSONDecodeError:
            response_dict = {}
            status_code = 500 if status_code != SUCCESS_STATUS_CODE else status_code
        return status_code, response_dict
