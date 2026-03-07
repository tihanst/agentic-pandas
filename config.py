from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMConfig(BaseSettings):
    provider: str 
    llm_name: str  
    llm_api_key: str | None = None
    is_local: bool 
    llm_endpoint: str 
    output_path: str 

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

class FilePathConfig(BaseSettings):
    output_path: str
    markdown_path: str
    history_path: str
    input_path: str
    html_path: str

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')