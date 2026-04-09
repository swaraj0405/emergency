from app.models.reward import GraderResult


MIN_SCORE = 0.001
MAX_SCORE = 0.999


def _norm_margin(travel_time: float, critical_limit: float) -> float:
    if critical_limit <= 0:
        return 0.0
    return max(0.0, min(1.0, (critical_limit - travel_time) / critical_limit))


def grade_task(
    task_id: str,
    difficulty: str,
    objective: str,
    trajectory: list[dict],
) -> GraderResult:
    """
    Grade task based on arrival outcomes (accepted, partial, rejected),
    not probabilistic survival.
    """
    steps = max(1, len(trajectory))
    
    # Count successful outcomes (accepted or partial admission)
    successful_outcomes = sum(
        1.0 for t in trajectory 
        if t.get("outcome_status") in ["accepted", "partial"]
    )
    success_rate = successful_outcomes / steps
    
    # Specialization match rate
    specialization_rate = sum(
        1.0 for t in trajectory if t.get("specialization_match", False)
    ) / steps
    
    # Time efficiency (based on travel times)
    margin_rate = sum(
        _norm_margin(t.get("travel_time", 0.0), t.get("critical_limit", 1.0)) 
        for t in trajectory
    ) / steps if trajectory else 0.0

    # Penalty for repeated failures at same hospital
    repeat_failures = 0
    visited_by_status = {}
    for step in trajectory:
        hospital_id = step.get("hospital_id", "unknown")
        status = step.get("outcome_status", "rejected")
        
        if status == "rejected":
            if hospital_id in visited_by_status and visited_by_status[hospital_id] == "rejected":
                repeat_failures += 1
        visited_by_status[hospital_id] = status

    repeat_failure_penalty = min(1.0, repeat_failures / steps)

    # Suitability component (how well hospital matched patient)
    avg_suitability = sum(
        t.get("suitability_score", 0.5) for t in trajectory
    ) / steps
    
    # Adaptive penalty: worse when early rejections vs later recovery
    adaptability_bonus = 0.0
    if len(trajectory) >= 2:
        outcomes = [t.get("outcome_status") for t in trajectory]
        if "rejected" in outcomes[:-1] and outcomes[-1] in ["accepted", "partial"]:
            adaptability_bonus = 0.1

    base = (
        (success_rate * 0.45)
        + (avg_suitability * 0.20)
        + (margin_rate * 0.15)
        + (specialization_rate * 0.15)
        + (adaptability_bonus * 0.05)
        - (repeat_failure_penalty * 0.01)
    )

    if difficulty == "easy":
        threshold = 0.73
        score = min(1.0, base + 0.1)
    elif difficulty == "medium":
        threshold = 0.62
        score = base
    else:  # hard
        threshold = 0.53
        hard_bonus = 0.15 if success_rate >= 0.5 else (0.05 if success_rate > 0.0 else 0.0)
        score = min(1.0, base + hard_bonus)

    score = max(MIN_SCORE, min(MAX_SCORE, score))

    return GraderResult(
        task_id=task_id,
        difficulty=difficulty,
        objective=objective,
        score=score,
        passed=score >= threshold,
        criteria={
            "success_rate": round(success_rate, 4),
            "suitability_rate": round(avg_suitability, 4),
            "margin_rate": round(margin_rate, 4),
            "specialization_rate": round(specialization_rate, 4),
            "repeat_failure_penalty": round(repeat_failure_penalty, 4),
            "adaptability_bonus": round(adaptability_bonus, 4),
            "threshold": threshold,
        },
    )
