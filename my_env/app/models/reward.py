from typing import Literal

from pydantic import BaseModel, Field


class RewardBreakdown(BaseModel):
    survival_component: float = Field(gt=0.0, lt=1.0)
    time_efficiency_component: float = Field(gt=0.0, lt=1.0)
    specialization_component: float = Field(gt=0.0, lt=1.0)
    delay_penalty: float = Field(gt=0.0, lt=1.0)


class RewardModel(BaseModel):
    value: float = Field(gt=0.0, lt=1.0)
    breakdown: RewardBreakdown


class GraderResult(BaseModel):
    task_id: Literal["acde_easy", "acde_medium", "acde_hard"]
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    score: float = Field(gt=0.0, lt=1.0)
    passed: bool
    criteria: dict[str, float] = Field(default_factory=dict)


class StepInfo(BaseModel):
    last_action_error: str | None = None
    task_id: Literal["acde_easy", "acde_medium", "acde_hard"]
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    progress_score: float = Field(gt=0.0, lt=1.0)
    reward_model: RewardModel
    grader: GraderResult | None = None
    outcome: dict[str, str] | None = None
