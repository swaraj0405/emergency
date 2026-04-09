from typing import Literal

from pydantic import BaseModel, Field, field_validator


# Strict boundaries for all score-like fields.
# The external validator requires scores strictly between 0 and 1
# (not 0.0 and not 1.0), so we clamp all values to [0.001, 0.999].
_FLOOR = 0.001
_CEIL = 0.999


def _clamp(v: float) -> float:
    """Clamp a score to the open interval (0, 1)."""
    return max(_FLOOR, min(_CEIL, v))


class RewardBreakdown(BaseModel):
    survival_component: float = Field(ge=_FLOOR, le=_CEIL)
    time_efficiency_component: float = Field(ge=_FLOOR, le=_CEIL)
    specialization_component: float = Field(ge=_FLOOR, le=_CEIL)
    delay_penalty: float = Field(ge=_FLOOR, le=_CEIL)

    @field_validator(
        "survival_component",
        "time_efficiency_component",
        "specialization_component",
        "delay_penalty",
        mode="before",
    )
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return _clamp(v)


class RewardModel(BaseModel):
    value: float = Field(ge=_FLOOR, le=_CEIL)
    breakdown: RewardBreakdown

    @field_validator("value", mode="before")
    @classmethod
    def clamp_value(cls, v: float) -> float:
        return _clamp(v)


class GraderResult(BaseModel):
    task_id: Literal["acde_easy", "acde_medium", "acde_hard"]
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    score: float = Field(ge=_FLOOR, le=_CEIL)
    passed: bool
    criteria: dict[str, float] = Field(default_factory=dict)

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return _clamp(v)


class StepInfo(BaseModel):
    last_action_error: str | None = None
    task_id: Literal["acde_easy", "acde_medium", "acde_hard"]
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    progress_score: float = Field(ge=_FLOOR, le=_CEIL)
    reward_model: RewardModel
    grader: GraderResult | None = None
    outcome: dict[str, str] | None = None

    @field_validator("progress_score", mode="before")
    @classmethod
    def clamp_progress(cls, v: float) -> float:
        return _clamp(v)
