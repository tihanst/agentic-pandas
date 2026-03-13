from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices

class LLMConfig(BaseSettings):
    provider: str = Field(default=...) 
    llm_name: str = Field(default=...) 
    #llm_api_key: str = Field(default=...)
    llm_api_key: str | None = Field(default=None, validation_alias=AliasChoices("TOGETHER_API_KEY")) # Make sure to add other providers here 
    is_local: bool = Field(default=...)
    llm_endpoint: str = Field(default=...) 

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


class FilePathConfig(BaseSettings):
    output_path: str = Field(default=...)
    markdown_path: str = Field(default=...)
    history_path: str = Field(default=...)
    input_path: str = Field(default=...)
    html_path: str = Field(default=...)

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')