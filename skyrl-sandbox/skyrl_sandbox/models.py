"""Pydantic models for environment specifications."""

from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class ParsingConfig(BaseModel):
    """Configuration for parsing model output."""
    method: Literal["regex", "json_path"] = "regex"
    pattern: Optional[str] = None  # For regex method
    json_path: Optional[str] = None  # For json_path method


class RewardConfig(BaseModel):
    """Configuration for reward calculation."""
    method: Literal["exact_match", "regex_match", "numeric_tolerance"] = "exact_match"
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    partial_reward: Optional[float] = None  # For format-correct but wrong answer
    tolerance: Optional[float] = None  # For numeric_tolerance method


class FeedbackConfig(BaseModel):
    """Configuration for environment feedback messages."""
    on_incorrect: str = "Your answer '{answer}' is incorrect. Please try again."
    on_format_error: str = "Please provide your answer in the correct format."


class EnvironmentSpec(BaseModel):
    """Complete specification for a SkyRL environment."""
    env_name: str = Field(..., description="Name of the environment (e.g., 'multiply', 'text2sql')")
    env_type: Literal["single_turn", "multi_turn"] = "single_turn"
    max_turns: int = Field(default=3, description="Maximum turns for multi-turn environments")
    
    parsing: ParsingConfig
    reward: RewardConfig
    feedback: FeedbackConfig
    
    done_condition: Literal[
        "always_single_step",
        "max_turns_only", 
        "correct_or_max_turns"
    ] = "always_single_step"
    
    description: Optional[str] = Field(None, description="Optional description of the environment")


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
    metadata: Dict[str, Any] = {}


class ExportRequest(BaseModel):
    """Request to export environment code."""
    spec: EnvironmentSpec


class ExportResponse(BaseModel):
    """Response from export endpoint."""
    success: bool
    message: str
    files_created: list[str]

