"""LLM for agent"""

import os
import sys
from typing import List, Dict, Union

from litellm import completion, acompletion

from .message import Message

class LLM():
    
    def __init__(self, model_provider: str, model_name: str, model_endpoint: str, is_local: bool):

        self.provider = model_provider
        self.model = model_name
        self.endpoint = model_endpoint
        self.is_local = is_local


    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider}, model={self.model}, endpoint={self.endpoint}, is_local={self.is_local})"


    async def acompletion_call(self, messages: List[Dict[str, Union[str, Message]]]):
        
        if not self.is_local:
            try:
                return await acompletion(self.endpoint, messages) # Note litellm.completion for together requires endpoint as model name
            except Exception as e:
                raise RuntimeError(f"LiteLLM completion failed for provider '{self.provider}' with details {repr(self)}. Ensure it is supported by LiteLLM, that the endpoint is correctly specified, and the appropriate API key is set in your environment.\n\nOriginal error: {e}\n")

        else:
            try:
                return await acompletion(model=self.model,
                              messages=messages,
                              api_base=self.endpoint)
            except Exception as e:
                raise RuntimeError(f"LiteLLM completion failed for local model '{self.provider} with details {repr(self)}. Check LiteLLM documentation for error:\n\n{e}\n")

    