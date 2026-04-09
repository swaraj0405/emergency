from typing import Literal

from pydantic import BaseModel, Field


class LearningEntry(BaseModel):
    success: int = Field(default=0, ge=0)
    fail: int = Field(default=0, ge=0)
    avg: float = Field(default=0.0, ge=0.0, le=1.0)
    accepted: int = Field(default=0, ge=0)
    rejected: int = Field(default=0, ge=0)


class HospitalValidationDetails(BaseModel):
    """Hidden validation checks performed after ambulance arrives at hospital"""
    icu_available: bool
    doctor_available: bool
    equipment_functional: bool
    overload_status: Literal["clear", "moderate", "severe"]
    patient_suitability: float = Field(ge=0.0, le=1.0)  # 0=unsuitable, 1=ideal


class ArrivalOutcome(BaseModel):
    """Result of hospital arrival attempt"""
    status: Literal["accepted", "partial", "rejected"]
    reason: str
    validation_details: HospitalValidationDetails | None = None
    reward_modifier: float = Field(default=1.0, ge=0.0, le=1.5)
    terminal: bool = False


class HospitalState(BaseModel):
    hospital_id: str
    distance_km: float = Field(ge=0)
    icu_display: Literal["unknown", "available"]
    icu_actual: bool
    specialization: Literal["cardiac", "trauma", "general"]
    traffic: Literal["low", "medium", "high"]


class EnvState(BaseModel):
    episode_id: int
    seed: int
    task_id: Literal["acde_easy", "acde_medium", "acde_hard"]
    task_objective: str
    scenario_type: Literal["medical", "accident", "fire"]
    scenario_name: str
    scenario_difficulty: Literal["easy", "medium", "hard"]
    patient_condition: str
    required_specialization: Literal["cardiac", "trauma", "general"]
    initial_critical_time_limit_minutes: float = Field(gt=0)
    critical_time_limit_minutes: float = Field(gt=0)
    step: int = Field(ge=1)
    max_steps: int = 3
    hospitals: list[HospitalState]
    selected_hospital_id: str | None = None
    done: bool = False
    final_outcome: Literal["SUCCESS", "FAILURE"] | None = None
    final_score: float = Field(default=0.001, ge=0.001, le=0.999)
    reward: float = Field(default=0.001, ge=0.001, le=0.999)
    ambulance_status: Literal["en_route", "in_transit", "arrived", "admitted", "rerouting"] = "en_route"
    current_location_context: str = "incident_site"
    visited_hospitals: list[str] = Field(default_factory=list)
    failed_hospitals: list[str] = Field(default_factory=list)
    recent_failed_hospitals: list[str] = Field(default_factory=list)
    failed_reasons: dict[str, str] = Field(default_factory=dict)
    total_time_spent_minutes: float = Field(default=0.0, ge=0.0)
    rerouting_reason: str | None = None
    # New fields for arrival tracking
    last_arrival_outcome: ArrivalOutcome | None = None
    accepted_hospital_id: str | None = None
    explanation: list[str] = Field(default_factory=list)
    memory: dict[str, LearningEntry] = Field(default_factory=dict)
