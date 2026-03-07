"""LLM for agent"""

import os
from typing import List, Dict, Any

from litellm import completion

from message import Message

class LLM():
    
    def __init__(self, model_provider: str, model_api_key: str | None, model_name: str, model_endpoint: str, is_local: bool, **kwargs):

        self.provider = model_provider
        self.model = model_name
        self.key = model_api_key if model_api_key else None
        self.endpoint = model_endpoint
        self.is_local = is_local
        self.extra_args: Dict[str, Any] = kwargs

        if self.provider == 'together':
            os.environ["TOGETHER_API_KEY"] = self.key
        
        if self.is_local:
            pass

    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider}, model={self.model}, endpoint={self.endpoint}, is_local={self.is_local})"


    def completion_call(self, messages: List[Message]):
        
        if self.provider == 'together':
            return completion(self.endpoint, messages)
        
        elif self.provider == 'ollama':
            return completion(model=self.model,
                              messages=messages,
                              api_base=self.endpoint)
        else:
            print(f"Model {print(self)} not recognized")
        
    