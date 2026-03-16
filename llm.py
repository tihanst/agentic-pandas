"""LLM for agent"""

import os
import sys
from typing import List, Dict, Union

from litellm import completion

from message import Message

class LLM():
    
    def __init__(self, model_provider: str, model_name: str, model_endpoint: str, is_local: bool):

        self.provider = model_provider
        self.model = model_name
        #self.key = model_api_key if model_api_key else None
        self.endpoint = model_endpoint
        self.is_local = is_local

        # Do not need as Litellm finds keys automatically 
        # if self.provider == 'together':
        #     try:
        #         self.key = os.environ["TOGETHER_API_KEY"]
        #     except KeyError:
        #         print(f"KeyError no TOGETHER_API_KEY\nSet environment variable or choose another LLM provider\n")
        #         raise

    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider}, model={self.model}, endpoint={self.endpoint}, is_local={self.is_local})"


    def completion_call(self, messages: List[Dict[str, Union[str, Message]]]):
        
        if not self.is_local:
            try:
                return completion(self.endpoint, messages) # Note litellm.completion for together requires endpoint as model name
            except Exception as e:
                raise RuntimeError(f"LiteLLM completion failed for provider '{self.provider}'. Ensure it is supported by LiteLLM, that the endpoint is correctly specified, and the appropriate API key is set in your environment.\n\nOriginal error: {e}")

        elif self.provider == 'ollama':
            return completion(model=self.model,
                              messages=messages,
                              api_base=self.endpoint)
        else:
            raise ValueError(f"Model {self} not recognized in LLM call.")

    