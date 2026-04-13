import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator

_ENV_FILE = Path(__file__).parent / ".env"

# Maps the PROVIDER value in .env to the env var name LiteLLM expects
_PROVIDER_KEY_ENV_VARS: dict[str, str] = {
    "together": "TOGETHER_API_KEY",
    "together_ai": "TOGETHER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "replicate": "REPLICATE_API_KEY",
    "google_gemini": "GEMINI_API_KEY" 
}

class LLMConfig(BaseSettings):
    provider: str = Field(default=...)
    llm_name: str = Field(default=...)
    is_local: bool = Field(default=...)
    llm_endpoint: str = Field(default=...)
    llm_api_key: str | None = Field(default=None)

    @model_validator(mode='after')
    def inject_api_key(self) -> 'LLMConfig':
        if self.llm_api_key:
            env_var = _PROVIDER_KEY_ENV_VARS.get(self.provider.lower())
            if env_var:
                os.environ[env_var] = self.llm_api_key
        return self

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra='ignore')


class FilePathConfig(BaseSettings):
    top_level_output_path: Path = Field(default=...)
    output_path: Path | None = None
    html_path: Path | None = None
    markdown_path: Path | None = None
    history_path: Path | None = None
    input_path: Path | None = None

    @model_validator(mode='after')
    def set_derived_paths(self) -> 'FilePathConfig':
        if not self.top_level_output_path.is_absolute():
            self.top_level_output_path = (Path(__file__).parent / self.top_level_output_path).resolve()
        if not self.output_path:
            self.output_path = self.top_level_output_path / "output_files"
        if not self.html_path:
            self.html_path = self.top_level_output_path / "html_files"
        if not self.markdown_path:
            self.markdown_path = self.top_level_output_path / "markdown_files"
        if not self.history_path:
            self.history_path = self.top_level_output_path / "history"
        if not self.input_path:
            self.input_path = self.top_level_output_path / "data_input_files"
        return self

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra='ignore')