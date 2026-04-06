from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator

class LLMConfig(BaseSettings):
    provider: str = Field(default=...)
    llm_name: str = Field(default=...)
    #llm_api_key: str | None = Field(default=None, validation_alias=AliasChoices("TOGETHER_API_KEY")) # Not needed as LiteLLM finds API key automatically
    is_local: bool = Field(default=...)
    llm_endpoint: str = Field(default=...)

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


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

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')