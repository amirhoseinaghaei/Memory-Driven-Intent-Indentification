import requests
import time
from typing import List, Optional


class Embed:
    """
    TensorBlock / OpenAI-compatible embedding client.

    Supports:
      - embed_query(text)       -> List[float]
      - embed_documents(texts)  -> List[List[float]]
    """

    def __init__(
        self,
        settings,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.base_url = settings.EMBEDDING_API_BASE.rstrip("/")
        self.api_key = settings.API_KEY
        self.model = settings.EMBEDDING_MODEL
        self.timeout = timeout
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )
        self.session.trust_env = False

        self.embed_url = f"{self.base_url}/v1/embeddings"
        print(f"Initialized Embed client with model '{self.model}' at '{self.embed_url}'")
    def _warmup(self):
        try:
            self.embed_query("warmup")
        except Exception:
            pass

    def _post(self, payload: dict) -> dict:
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(
                    self.embed_url,
                    json=payload,
                    timeout=self.timeout,
                )

                if not resp.ok:
                    try:
                        err_json = resp.json()
                    except Exception:
                        err_json = resp.text
                    raise RuntimeError(
                        f"Embedding API error {resp.status_code}: {err_json}"
                    )

                return resp.json()

            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                else:
                    raise RuntimeError(
                        f"Embedding request failed after {self.max_retries + 1} attempts"
                    ) from last_err

        raise AssertionError("unreachable")

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single string.
        """
        try: 
            payload = {
                "model": self.model,
                "input": text,
            }
            data = self._post(payload)
            return data["data"][0]["embedding"]
        except Exception as exc:
            print(f"Error in embed_query: {exc}")
            return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of strings.
        """
        if not texts:
            return []

        payload = {
            "model": self.model,
            "input": [str(t) for t in texts],
        }

        data = self._post(payload)
        return [item["embedding"] for item in data["data"]]