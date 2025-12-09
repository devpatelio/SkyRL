"""Pydantic models for the SkyRL sandbox."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

DEFAULT_LLM_PROMPT = (
    "You are a strict automated judge for reinforcement learning rewards.\n"
    "Given the model response:\n"
    "{model_output}\n\n"
    "and the ground truth reference answer:\n"
    "{ground_truth}\n\n"
    "Return a JSON object with keys `score` (between 0 and 1) and "
    "`explanation`. Score 1 only if the response is fully correct."
)


class ParsingConfig(BaseModel):
    """Configuration for parsing model output."""

    method: Literal["regex", "json_path"] = "regex"
    pattern: Optional[str] = None
    json_path: Optional[str] = None


class LLMVerifierConfig(BaseModel):
    """Settings for optional LLM-based reward verification."""

    provider: Literal["mock", "openai"] = "mock"
    model: str = "gpt-4o-mini"
    prompt_template: str = DEFAULT_LLM_PROMPT
    temperature: float = 0.0
    max_output_tokens: int = 200
    success_threshold: float = 0.5
    response_format: Literal["score", "boolean"] = "score"
    success_keywords: List[str] = Field(default_factory=lambda: ["yes", "correct", "true"])
    api_key_env: str = "OPENAI_API_KEY"
    api_base: Optional[str] = None


class RewardCreationConfig(BaseModel):
    """How to transform outputs into a reward signal before applying schemes."""

    method: Literal["parsed_answer_rule", "json_path_rule", "llm_verifier"] = "parsed_answer_rule"
    rule_type: Literal["exact_match", "regex_match", "numeric_tolerance"] = "exact_match"
    numeric_tolerance: Optional[float] = None
    regex_pattern: Optional[str] = None
    json_path: Optional[str] = None
    json_success_values: List[Any] = Field(default_factory=list)
    json_threshold: Optional[float] = None
    llm: Optional[LLMVerifierConfig] = None


class RewardSchemeConfig(BaseModel):
    """Final mapping from reward signal to numeric reward."""

    scheme: Literal["binary", "partial", "dense"] = "binary"
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    partial_reward: Optional[float] = None
    format_error_reward: float = 0.0


class RewardPipeline(BaseModel):
    """Legacy-friendly wrapper that ties creation + scheme."""

    creation: RewardCreationConfig = Field(default_factory=RewardCreationConfig)
    scheme: RewardSchemeConfig = Field(default_factory=RewardSchemeConfig)

    @model_validator(mode="before")
    @classmethod
    def support_legacy_schema(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        if "creation" in value or "scheme" in value:
            return value

        # Legacy payload looked like RewardConfig
        legacy = value
        creation = {
            "method": "parsed_answer_rule",
            "rule_type": legacy.get("method", "exact_match"),
            "numeric_tolerance": legacy.get("tolerance"),
        }
        scheme = {
            "scheme": "partial" if legacy.get("partial_reward") is not None else "binary",
            "correct_reward": legacy.get("correct_reward", 1.0),
            "incorrect_reward": legacy.get("incorrect_reward", 0.0),
            "partial_reward": legacy.get("partial_reward"),
        }
        return {"creation": creation, "scheme": scheme}


class ToolConfig(BaseModel):
    """Configuration for a single tool."""

    tool_type: Literal["python", "sql", "search"] = "python"
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)


class PythonToolConfig(BaseModel):
    """Configuration for Python code execution tool."""

    timeout: float = 15.0
    allowed_imports: List[str] = Field(default_factory=lambda: ["math", "numpy", "pandas"])
    restrict_file_operations: bool = True


class SQLToolConfig(BaseModel):
    """Configuration for SQL execution tool."""

    db_file_path: Optional[str] = None
    timeout: float = 30.0
    read_only: bool = True
    max_result_rows: int = 1000


class SearchToolConfig(BaseModel):
    """Configuration for search tool."""

    search_url: str = "http://127.0.0.1:8000/retrieve"
    max_results: int = 10
    timeout: float = 10.0


class ToolsConfig(BaseModel):
    """Configuration for all tools in an environment."""

    enabled: bool = False
    python: Optional[PythonToolConfig] = None
    sql: Optional[SQLToolConfig] = None
    search: Optional[SearchToolConfig] = None

    def get_enabled_tools(self) -> List[str]:
        """Return list of enabled tool types."""
        enabled_tools = []
        if self.python:
            enabled_tools.append("python")
        if self.sql:
            enabled_tools.append("sql")
        if self.search:
            enabled_tools.append("search")
        return enabled_tools


class FeedbackConfig(BaseModel):
    """Configuration for environment feedback messages."""

    on_incorrect: str = "Your answer '{answer}' is incorrect. Please try again."
    on_format_error: str = "Please provide your answer in the correct format."


class EnvironmentSpec(BaseModel):
    """Complete specification for a SkyRL environment."""

    model_config = ConfigDict(populate_by_name=True)

    env_name: str = Field(
        ...,
        description="Name of the environment (e.g., 'multiply', 'text2sql')",
    )
    env_type: Literal["single_turn", "multi_turn"] = "single_turn"
    max_turns: int = Field(default=3, description="Maximum turns for multi-turn environments")

    parsing: ParsingConfig
    reward: RewardPipeline = Field(
        default_factory=RewardPipeline,
        validation_alias=AliasChoices("reward", "reward_pipeline"),
        serialization_alias="reward",
    )
    feedback: FeedbackConfig

    done_condition: Literal["always_single_step", "max_turns_only", "correct_or_max_turns"] = "always_single_step"

    description: Optional[str] = Field(None, description="Optional description of the environment")

    tools: ToolsConfig = Field(
        default_factory=ToolsConfig, description="Configuration for tools available to the agent"
    )


class PreviewRequest(BaseModel):
    """Request for previewing reward calculation."""

    spec: EnvironmentSpec
    model_output: str
    ground_truth: Any
    current_turn: int = 1


class PreviewResponse(BaseModel):
    """Response from preview endpoint."""

    parsed_answer: Optional[str]
    is_correct: bool
    reward: float
    done: bool
    feedback: Optional[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExportRequest(BaseModel):
    """Request to export environment code."""

    spec: EnvironmentSpec


class ExportResponse(BaseModel):
    """Response from export endpoint."""

    success: bool
    message: str
    files_created: List[str]


# Dataset builder models -----------------------------------------------------


class ChatMessage(BaseModel):
    """Single message inside a prompt conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class RewardSpecConfig(BaseModel):
    """Dataset-level reward specification."""

    method: Literal["rule", "reward_model"] = "rule"
    ground_truth: Optional[str] = None
    scorer_model: Optional[str] = None
    scorer_prompt: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class DatasetEntry(BaseModel):
    """One dataset sample in SkyRL format."""

    data_source: Optional[str] = None
    prompt: List[ChatMessage]
    env_class: Optional[str] = None
    reward_spec: Optional[RewardSpecConfig] = None
    extra_info: Dict[str, Any] = Field(default_factory=dict)


class DatasetDefaults(BaseModel):
    """Defaults that apply to entries if omitted."""

    data_source: str = "custom"
    env_class: str = "custom_env"
    reward_spec: RewardSpecConfig = Field(default_factory=RewardSpecConfig)
    extra_info: Dict[str, Any] = Field(default_factory=dict)


class DatasetSplit(BaseModel):
    """Named split such as train/validation/test."""

    name: str
    entries: List[DatasetEntry] = Field(default_factory=list)


class DatasetSpec(BaseModel):
    """Overall dataset definition."""

    dataset_name: str
    description: Optional[str] = None
    defaults: DatasetDefaults = Field(default_factory=DatasetDefaults)
    splits: List[DatasetSplit] = Field(default_factory=list)


class DatasetExportRequest(BaseModel):
    """Export dataset into json/parquet artifacts."""

    spec: DatasetSpec
    format: Literal["json", "parquet"] = "json"


class DatasetExportResponse(BaseModel):
    """Export result metadata."""

    artifact_dir: str
    files: List[Dict[str, Any]]


class DatasetImportRequest(BaseModel):
    """Import dataset from external source."""

    source: Literal["file", "huggingface", "skyrl"] = "file"
    path: str
    sample_size: Optional[int] = None
    split: Optional[str] = None


# Training models ------------------------------------------------------------


class TrainingLaunchRequest(BaseModel):
    """Request payload for launching a training run."""

    env_name: str
    run_name: str
    train_data: List[str]
    val_data: List[str]
    hydra_overrides: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    use_uv: bool = True
    environment_overrides: Dict[str, Any] = Field(default_factory=dict)
    venv_path: Optional[str] = Field(
        default="~/src/venv/skyrl/bin/activate",
        description="Path to virtual environment activation script. If provided, will be activated before running the training command.",
    )


class TrainingRunInfo(BaseModel):
    """Serialized info about a training run."""

    id: str
    env_name: str
    run_name: str
    status: Literal["pending", "running", "success", "error", "stopped"]
    command: List[str]
    log_path: str
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    notes: Optional[str] = None


class TrainingLogsResponse(BaseModel):
    """Chunk of logs for a given run."""

    run_id: str
    tail: int = 200
    lines: List[str]
