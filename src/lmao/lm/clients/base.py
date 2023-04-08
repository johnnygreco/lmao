import os
import re
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

__all__ = ["BaseClientResponse", "BaseClient", "ChatHistory", "SUCCESS_STATUS_CODE"]
SUCCESS_STATUS_CODE = 200


@dataclass
class BaseClientResponse:
    text: Optional[str]
    raw_response: dict
    status_code: int

    def __repr__(self):
        repr = "\n".join([f"{k}: {v}" for k, v in self.__dict__.items()])
        repr = re.sub(r"^", " " * 4, repr, 0, re.M)
        return f"{self.__class__.__name__}({{\n{repr}\n}})"


class ChatHistory(ABC):
    def __init__(self, length: int = 10):
        self.length = length
        self._messages: Deque = deque(maxlen=length)

    @abstractmethod
    def append(self, *args, **kwargs):
        pass

    @abstractmethod
    def check_message_format(self, message):
        pass

    def clear(self):
        self._messages.clear()

    def __iter__(self):
        return iter(self._messages)

    def __len__(self):
        return len(self._messages)

    def __repr__(self):
        repr = "\n".join([str(m) for m in self._messages])
        repr = "\n" + re.sub(r"^", " " * 4, repr, 0, re.M) + "\n" if len(self._messages) > 0 else ""
        return f"{self.__class__.__name__}([{repr}])"


class LM(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> BaseClientResponse:
        pass


class BaseClient(LM, ABC):
    base_url: str = "none"
    api_env_name: str = "none"

    #  If the backoff_factor is 0.1, then sleep() will sleep for [0.0s, 0.2s, 0.4s, …] between retries.
    RETRY_BACKOFF_FACTOR: float = 0.1
    RETRY_STATUS_CODES: List[int] = [429, 500, 502, 503, 504]

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 5):
        self.max_retries = max_retries
        self.__api_key = api_key or os.environ.get(self.api_env_name)
        if self.__api_key is None:
            raise ValueError("You must provide an API key or set api_env_name to initialize an LM Client.")
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
