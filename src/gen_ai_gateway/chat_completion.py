import httpx
from openai import OpenAI
from abc import ABC
from src.config.config import settings


class ChatCompletion(ABC):

   
    def __init__(self, settings):
        self.settings = settings

        self.model_name = settings.NO_THINK_MODEL_NAME
        self.temperature = settings.TEMPERATURE
        self.max_tokens = settings.MAX_OUTPUT_TOKEN

        self.client = OpenAI(
            api_key=settings.API_KEY,
            base_url=settings.API_BASE,
            http_client=httpx.Client(verify=False)
        )
    def create_response(self, message, max_tokens = 4096, **kwargs):
        return  self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            temperature=self.temperature,
            stream=False,
            **kwargs
        )

