#!/usr/bin/env python3
"""Local agent runner for EmergencyEnv.

This script acts as an agent only:
- reset env
- choose action from observation
- step env
- log trajectory
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from datetime import datetime, timezone

from app.environment.core import EmergencyEnv
from app.models.action import Action

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback for missing optional dependency
    OpenAI = None

TASK_ORDER = ["acde_easy", "acde_medium", "acde_hard"]
LEVEL_TO_TASK = {
    "low": "acde_easy",
    "medium": "acde_medium",
    "high": "acde_hard",
}
RANDOM_LEVELS = ("medium", "high")
RANDOM_LEVEL_WEIGHTS = (0.25, 0.75)
BASE_SPEED_KMH = 60.0
TRAFFIC_FACTOR = {"low": 1.0, "medium": 0.6, "high": 0.3}
LEARNING_ARCHIVE_PATH = Path(__file__).resolve().parent / "data" / "learning_archive.json"
LEARNING_ARCHIVE_VERSION = 2
DEFAULT_API_BASE_URL = "https://api-inference.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
REQUIRED_ENV_VARS = ("HF_TOKEN",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EmergencyEnv agent runner")
    parser.add_argument("--mode", choices=["single", "full"], default="single")
    parser.add_argument("--task", choices=TASK_ORDER, default=None)
    parser.add_argument("--level", choices=["low", "medium", "high"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--train-episodes", type=int, default=0)
    parser.add_argument("--train-same-seed", action="store_true")
    parser.add_argument(
        "--memory-file",
        default=str(Path(__file__).resolve().parent / "data" / "learning_memory.json"),
    )
    return parser.parse_args()


def emit_structured(tag: str, payload: dict) -> None:
    print(f"[{tag}] " + json.dumps(payload, ensure_ascii=True, separators=(",", ":")))


def runtime_llm_config() -> dict[str, str]:
    return {
        "API_BASE_URL": os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).strip(),
        "MODEL_NAME": os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip(),
        "HF_TOKEN": os.getenv("HF_TOKEN", "").strip(),
    }


def require_llm_config() -> tuple[object, str]:
    config = runtime_llm_config()
    missing = [name for name, value in config.items() if not value]
    if missing:
        raise SystemExit(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Set HF_TOKEN before running inference.py"
        )
    if OpenAI is None:
        raise SystemExit("openai package is required for inference.py LLM rationale generation.")

    client = OpenAI(base_url=config["API_BASE_URL"], api_key=config["HF_TOKEN"], timeout=8.0)
    return client, config["MODEL_NAME"]


def llm_rationale(
    client: object,
    model_name: str,
    observation: dict,
    chosen: dict,
    strategy: str,
) -> str:
    fallback = (
        f"Selected {chosen['hospital_id']} by {strategy}; "
        f"score={chosen['policy_score']:.3f}, traffic={chosen['traffic']}, icu={chosen['icu']}"
    )
    try:
        prompt = (
            "You are an emergency routing agent. Return one short sentence rationale "
            "for the selected hospital. Keep it under 25 words.\n"
            f"task={observation.get('task_id')} difficulty={observation.get('scenario_difficulty')} "
            f"step={observation.get('step')} patient={observation.get('patient_condition')} "
            f"required={observation.get('required_specialization')} "
            f"selected={chosen['hospital_id']} score={chosen['policy_score']:.3f} "
            f"distance={chosen['distance_km']:.1f}km traffic={chosen['traffic']} icu={chosen['icu']} "
            f"strategy={strategy}"
        )
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Generate concise emergency triage rationale."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=60,
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            return fallback
        return " ".join(text.split())[:180]
    except Exception:
        return fallback


def normalize_seed(raw_value: int | str) -> int:
    """Normalize arbitrary numeric/text input into a deterministic positive seed."""
    if isinstance(raw_value, int):
        value = raw_value
    else:
        text = str(raw_value).strip()
        try:
            value = int(text)
        except ValueError:
            # Deterministic fallback for non-numeric input.
            value = sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))

    normalized = abs(value) % 1_000_000_000
    return normalized if normalized != 0 else 202601


def ask_seed_if_missing(seed: int | None) -> int:
    if seed is not None:
        return normalize_seed(seed)
    # No CLI seed means a fresh randomized run.
    return normalize_seed(random.SystemRandom().randint(1, 999_999_999))


def ask_level_if_missing(level: str | None) -> str:
    if level in LEVEL_TO_TASK:
        return level
    # No CLI level means pick a random non-easy difficulty.
    return random.choices(
        RANDOM_LEVELS,
        weights=RANDOM_LEVEL_WEIGHTS,
        k=1,
    )[0]


def append_trajectory_log(entry: dict) -> None:
    path = Path(__file__).resolve().parent / "data" / "trajectory_history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=True) + "\n")


def load_learning_archive() -> dict:
    LEARNING_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LEARNING_ARCHIVE_PATH.exists():
        return {"version": LEARNING_ARCHIVE_VERSION, "profiles": {}, "episodes": []}

    try:
        payload_text = LEARNING_ARCHIVE_PATH.read_text(encoding="utf-8-sig").strip()
        payload = json.loads(payload_text) if payload_text else {}
    except json.JSONDecodeError:
        return {"version": LEARNING_ARCHIVE_VERSION, "profiles": {}, "episodes": []}

    if not isinstance(payload, dict):
        return {"version": LEARNING_ARCHIVE_VERSION, "profiles": {}, "episodes": []}

    if payload.get("version") != LEARNING_ARCHIVE_VERSION:
        return {
            "version": LEARNING_ARCHIVE_VERSION,
            "profiles": {},
            "episodes": payload.get("episodes", [])[-500:] if isinstance(payload.get("episodes", []), list) else [],
        }

    payload.setdefault("version", LEARNING_ARCHIVE_VERSION)
    payload.setdefault("profiles", {})
    payload.setdefault("episodes", [])
    return payload


def save_learning_archive(archive: dict) -> None:
    LEARNING_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEARNING_ARCHIVE_PATH.write_text(json.dumps(archive, indent=2, ensure_ascii=True), encoding="utf-8")


def profile_key(seed: int, task_id: str) -> str:
    return f"{seed}|{task_id}"


def _merge_step_stats(primary: dict, secondary: dict) -> dict:
    merged: dict = {}
    for step_key in set(primary.keys()) | set(secondary.keys()):
        merged[step_key] = {}
        step_primary = primary.get(step_key, {})
        step_secondary = secondary.get(step_key, {})
        for hospital_id in set(step_primary.keys()) | set(step_secondary.keys()):
            a = step_primary.get(hospital_id, {})
            b = step_secondary.get(hospital_id, {})
            count = int(a.get("count", 0)) + int(b.get("count", 0))
            accepted = int(a.get("accepted", 0)) + int(b.get("accepted", 0))
            partial = int(a.get("partial", 0)) + int(b.get("partial", 0))
            rejected = int(a.get("rejected", 0)) + int(b.get("rejected", 0))
            total_reward = float(a.get("total_reward", 0.0)) + float(b.get("total_reward", 0.0))
            merged[step_key][hospital_id] = {
                "count": count,
                "success": int(a.get("success", 0)) + int(b.get("success", 0)),
                "accepted": accepted,
                "partial": partial,
                "rejected": rejected,
                "total_reward": total_reward,
                "avg_reward": (total_reward / max(1, count)),
                "success_rate": (accepted / max(1, count)),
                "last_status": a.get("last_status") or b.get("last_status"),
                "last_reason": a.get("last_reason") or b.get("last_reason"),
            }
    return merged


def build_learning_profile(
    archive: dict,
    seed: int,
    task_id: str,
    required_specialization: str | None = None,
) -> dict | None:
    profiles = archive.get("profiles", {})
    key = profile_key(seed, task_id)
    exact = profiles.get(key)
    if not exact:
        return None

    # Strict scope: learn only from same seed + same level/task.
    return {
        "attempts": int(exact.get("attempts", 0)),
        "best_score": float(exact.get("best_score", 0.0)),
        "best_actions": list(exact.get("best_actions", [])),
        "step_stats": exact.get("step_stats", {}),
        "best_scenario_name": exact.get("best_scenario_name"),
        "last_scenario_name": exact.get("last_scenario_name"),
        "source": "exact-only",
    }


def _difficulty_policy_params(difficulty: str) -> tuple[float, float]:
    if difficulty == "easy":
        return 0.07, 0.18
    if difficulty == "medium":
        return 0.16, 0.32
    return 0.26, 0.44


def _sample_softmax(candidates: list[dict], key: str, temperature: float, rng: random.Random) -> dict:
    logits = [item[key] / max(temperature, 1e-6) for item in candidates]
    max_logit = max(logits)
    exps = [math.exp(v - max_logit) for v in logits]
    total = sum(exps)
    probs = [e / total for e in exps]

    roll = rng.random()
    cdf = 0.0
    for item, prob in zip(candidates, probs):
        cdf += prob
        if roll <= cdf:
            return item
    return candidates[-1]


def memory_score_for_hospital(
    hospital_id: str,
    memory_snapshot: dict,
    learning_profile: dict | None = None,
    step_number: int | None = None,
) -> float:
    entry = memory_snapshot.get(hospital_id)
    if not entry:
        return 0.5

    success = int(entry.get("accepted", entry.get("success", 0)))
    fail = int(entry.get("rejected", entry.get("fail", 0)))
    avg = float(entry.get("avg", 0.0))
    total = success + fail
    if total <= 0:
        return 0.5

    success_rate = success / total
    # Fix 3: reliability-first memory scoring.
    value = (0.6 * success_rate) + (0.4 * avg)
    recent_failed = False

    if learning_profile and step_number is not None:
        step_stats = learning_profile.get("step_stats", {}).get(str(step_number), {})
        hospital_stats = step_stats.get(hospital_id)
        if hospital_stats:
            step_avg = float(hospital_stats.get("avg_reward", 0.0))
            step_success = float(hospital_stats.get("success_rate", 0.0))
            step_count = int(hospital_stats.get("count", 0))
            value += min(0.20, (step_avg * 0.10) + (step_success * 0.08) + min(step_count, 5) * 0.01)
            recent_failed = str(hospital_stats.get("last_status", "")).upper() == "REJECTED"

    if recent_failed:
        value -= 0.3

    return max(0.0, min(1.0, value))


def score_hospitals(observation: dict, learning_profile: dict | None = None) -> list[dict]:
    failed = set(observation.get("failed_hospitals", []))
    recent_failed = set(observation.get("recent_failed_hospitals", []))
    visited = set(observation.get("visited_hospitals", []))
    memory_snapshot = observation.get("memory_snapshot", {})
    previous_action = observation.get("previous_action")
    last_arrival = observation.get("last_arrival_outcome") or {}
    last_status = str(last_arrival.get("status", "")).lower()

    scored: list[dict] = []
    initial_limit = float(observation.get("initial_critical_time_limit_minutes", observation["critical_time_limit_minutes"]))
    remaining_time = float(observation.get("remaining_time_minutes", observation["critical_time_limit_minutes"]))
    urgency = 1.0 - min(1.0, max(0.0, remaining_time / max(initial_limit, 1e-6)))

    patient_condition = observation.get("patient_condition", "").lower()
    critical_patient = patient_condition in {"critical", "unstable"}
    required_specialization = str(observation.get("required_specialization", ""))
    scenario_name = str(observation.get("scenario_name", ""))
    step_number = int(observation.get("step", 1))
    difficulty = str(observation.get("scenario_difficulty", "medium"))
    attempts = int(learning_profile.get("attempts", 0)) if learning_profile else 0
    preferred_route = []
    if learning_profile:
        preferred_route = list(learning_profile.get("best_actions", []))

    for hospital in observation.get("hospitals", []):
        traffic_factor = TRAFFIC_FACTOR[hospital["traffic"]]
        speed_kmh = BASE_SPEED_KMH * traffic_factor
        travel_time = (hospital["distance_km"] / max(speed_kmh, 1e-6)) * 60.0

        distance_score = max(0.0, min(1.0, 1.0 - hospital["distance_km"] / 20.0))
        icu_score = 1.0 if hospital["icu"] == "available" else 0.55
        mem_score = memory_score_for_hospital(
            hospital["hospital_id"],
            memory_snapshot,
            learning_profile=learning_profile,
            step_number=step_number,
        )

        memory_scenario = ""
        if learning_profile:
            memory_scenario = str(
                learning_profile.get("best_scenario_name")
                or learning_profile.get("last_scenario_name")
                or ""
            )
        if memory_scenario and scenario_name and memory_scenario != scenario_name:
            mem_score *= 0.5

        spec_match = (
            hospital["specialization"] == observation["required_specialization"]
            or hospital["specialization"] == "general"
            or observation["required_specialization"] == "general"
        )
        exact_spec_match = hospital["specialization"] == observation["required_specialization"]
        general_fallback = (
            hospital["specialization"] == "general"
            and observation["required_specialization"] != "general"
        )

        rejected_penalty = 0.40 if hospital["hospital_id"] in failed else 0.0
        revisit_penalty = 0.14 if hospital["hospital_id"] in visited else 0.0
        partial_repeat_penalty = (
            0.32
            if last_status == "partial" and hospital["hospital_id"] == previous_action
            else 0.0
        )
        critical_unknown_penalty = 0.17 if critical_patient and hospital["icu"] == "unknown" else 0.03
        traffic_penalty = 0.10 if hospital["traffic"] == "high" else 0.04 if hospital["traffic"] == "medium" else 0.0
        if critical_patient and general_fallback:
            spec_penalty = {"easy": 0.08, "medium": 0.16, "hard": 0.26}.get(difficulty, 0.16)
            if attempts >= 5:
                spec_penalty += 0.06
        else:
            spec_penalty = 0.0
        spec_bonus = 0.16 if exact_spec_match else (0.08 if spec_match else 0.0)
        urgency_boost = urgency * (0.18 + max(0.0, 0.25 - travel_time / 100.0))
        step_route_bonus = 0.0
        if step_number - 1 < len(preferred_route) and preferred_route[step_number - 1] == hospital["hospital_id"]:
            step_route_bonus = 0.16

        score = (
            (icu_score * 0.30)
            + (distance_score * 0.18)
            + (traffic_factor * 0.14)
            + (mem_score * 0.24)
            + spec_bonus
            + urgency_boost
            + step_route_bonus
            - rejected_penalty
            - revisit_penalty
            - partial_repeat_penalty
            - spec_penalty
            - critical_unknown_penalty
            - traffic_penalty
        )

        if hospital["hospital_id"] == previous_action and last_status == "rejected":
            score *= 0.01

        if hospital["hospital_id"] in recent_failed:
            score *= 0.2

        if hospital["specialization"] != required_specialization:
            if patient_condition == "critical":
                score *= 0.15
            else:
                score *= 0.4
        elif patient_condition == "critical":
            score *= 1.5

        # Hard realism penalties to align policy scoring with validator outcomes.
        if hospital["specialization"] != required_specialization:
            score -= 0.6
        if critical_patient and hospital["icu"] == "unknown":
            score -= 0.5
        if critical_patient and hospital["traffic"] == "high":
            score -= 0.3

        # Confidence-style risk multiplier keeps risky options from looking deceptively good.
        risk_factor = 1.0
        if hospital["icu"] == "unknown":
            risk_factor *= 0.6
        if not spec_match:
            risk_factor *= 0.5
        if critical_patient and hospital["traffic"] == "high":
            risk_factor *= 0.7
        score *= risk_factor

        # Reduce memory dominance in final decision score.
        memory_weight = 0.15
        current_score_weight = 0.85
        if step_number == 1:
            memory_weight = 0.1
            current_score_weight = 0.9
        base_current_score = score
        confidence_score = max(0.0, min(1.0, base_current_score))
        effective_memory_score = mem_score
        in_best_route = hospital["hospital_id"] in preferred_route
        if in_best_route and confidence_score < 0.6:
            effective_memory_score = 0.0
        if confidence_score < 0.2:
            effective_memory_score = 0.0

        score = (current_score_weight * base_current_score) + (memory_weight * effective_memory_score)

        scored.append(
            {
                "hospital_id": hospital["hospital_id"],
                "icu": hospital["icu"],
                "distance_km": hospital["distance_km"],
                "traffic": hospital["traffic"],
                "specialization": hospital["specialization"],
                "travel_time": travel_time,
                "memory_score": mem_score,
                "policy_score": max(0.0, min(1.0, score)),
                "specialization_match": spec_match,
                "tie_break_score": (
                    (distance_score * 0.35)
                    + (traffic_factor * 0.35)
                    + (icu_score * 0.20)
                    + (0.10 if spec_match else 0.0)
                ),
            }
        )

    scored.sort(key=lambda item: item["policy_score"], reverse=True)
    if scored:
        min_score = min(item["policy_score"] for item in scored)
        max_score = max(item["policy_score"] for item in scored)
        spread = max_score - min_score
        if spread > 1e-9:
            for item in scored:
                normalized = (item["policy_score"] - min_score) / (spread + 1e-6)
                if normalized < 0.2:
                    jitter_seed = (
                        int(observation.get("seed", 0))
                        + (step_number * 131)
                        + sum(ord(ch) for ch in item["hospital_id"])
                    )
                    jitter_rng = random.Random(jitter_seed)
                    normalized *= jitter_rng.uniform(0.3, 0.7)
                item["policy_score"] = max(0.0, min(1.0, normalized))
        elif max_score > 0:
            for item in scored:
                normalized = item["policy_score"] / max_score
                if normalized < 0.2:
                    jitter_seed = (
                        int(observation.get("seed", 0))
                        + (step_number * 131)
                        + sum(ord(ch) for ch in item["hospital_id"])
                    )
                    jitter_rng = random.Random(jitter_seed)
                    normalized *= jitter_rng.uniform(0.3, 0.7)
                item["policy_score"] = max(0.0, min(1.0, normalized))
        else:
            tie_min = min(item.get("tie_break_score", 0.0) for item in scored)
            tie_max = max(item.get("tie_break_score", 0.0) for item in scored)
            tie_spread = tie_max - tie_min
            if tie_spread > 1e-9:
                for item in scored:
                    normalized = (item.get("tie_break_score", 0.0) - tie_min) / (tie_spread + 1e-6)
                    if normalized < 0.2:
                        jitter_seed = (
                            int(observation.get("seed", 0))
                            + (step_number * 131)
                            + sum(ord(ch) for ch in item["hospital_id"])
                        )
                        jitter_rng = random.Random(jitter_seed)
                        normalized *= jitter_rng.uniform(0.3, 0.7)
                    item["policy_score"] = max(0.0, min(1.0, normalized))
            else:
                for item in scored:
                    item["policy_score"] = 0.0

        # Remove hard-zero scores and normalize to probability-like values.
        for item in scored:
            if item["policy_score"] <= 0.0:
                jitter_seed = (
                    int(observation.get("seed", 0))
                    + (step_number * 173)
                    + sum(ord(ch) for ch in item["hospital_id"])
                )
                jitter_rng = random.Random(jitter_seed)
                if critical_patient and required_specialization != "general":
                    if item.get("specialization") == required_specialization:
                        item["policy_score"] = jitter_rng.uniform(0.08, 0.18)
                    else:
                        item["policy_score"] = jitter_rng.uniform(0.001, 0.01)
                else:
                    item["policy_score"] = jitter_rng.uniform(0.05, 0.15)

        total_score = sum(item["policy_score"] for item in scored)
        if total_score > 0:
            for item in scored:
                item["policy_score"] = item["policy_score"] / (total_score + 1e-6)
        else:
            uniform = 1.0 / len(scored)
            for item in scored:
                item["policy_score"] = uniform

        # Final clinical-priority pass: in critical non-general cases,
        # exact specialization should dominate unless unavailable.
        if critical_patient and required_specialization != "general":
            for item in scored:
                if item.get("specialization") == required_specialization:
                    item["policy_score"] *= 1.5
                else:
                    item["policy_score"] *= 0.15

            boosted_total = sum(item["policy_score"] for item in scored)
            if boosted_total > 0:
                for item in scored:
                    item["policy_score"] = item["policy_score"] / boosted_total

        for item in scored:
            raw_score = float(item["policy_score"])
            normalized_score = raw_score / (1.0 + abs(raw_score))
            # Keep a small floor so no action is fully eliminated from exploration.
            if normalized_score < 0.01:
                jitter_seed = (
                    int(observation.get("seed", 0))
                    + (step_number * 211)
                    + sum(ord(ch) for ch in item["hospital_id"])
                )
                jitter_rng = random.Random(jitter_seed)
                normalized_score = jitter_rng.uniform(0.01, 0.03)
            item["policy_score"] = normalized_score

        scored.sort(key=lambda item: item["policy_score"], reverse=True)

    for item in scored:
        item.pop("tie_break_score", None)
    return scored


def choose_hospital(
    scored: list[dict],
    observation: dict,
    rng: random.Random,
    learning_profile: dict | None = None,
) -> tuple[dict, str]:
    difficulty = observation.get("scenario_difficulty", "medium")
    epsilon, temperature = _difficulty_policy_params(difficulty)

    failed = set(observation.get("failed_hospitals", []))
    recent_failed = set(observation.get("recent_failed_hospitals", []))
    visited = set(observation.get("visited_hospitals", []))
    previous_action = observation.get("previous_action")
    selected_hospital_id = observation.get("selected_hospital_id")
    visited_sequence = observation.get("visited_hospitals", []) or []
    recent_hospital = previous_action or selected_hospital_id or (visited_sequence[-1] if visited_sequence else None)
    last_arrival = observation.get("last_arrival_outcome") or {}
    last_status = str(last_arrival.get("status", "")).lower()
    last_reason = str(last_arrival.get("reason", "")).lower()
    is_rerouting_phase = str(observation.get("ambulance_status", "")).lower() == "rerouting"

    # Cooldown logic: avoid recently failed hospitals first, then avoid visited when alternatives exist.
    candidates = [
        item
        for item in scored
        if item["hospital_id"] not in recent_failed and item["hospital_id"] not in visited
    ]
    if not candidates:
        candidates = [item for item in scored if item["hospital_id"] not in recent_failed]
    if not candidates:
        # Last-resort fallback: if every hospital has failed already, avoid immediate retry.
        candidates = list(scored)
        if (last_status == "rejected" or is_rerouting_phase) and recent_hospital:
            redirected = [item for item in candidates if item["hospital_id"] != recent_hospital]
            if redirected:
                candidates = redirected

    step_number = int(observation.get("step", 1))
    attempts = int(learning_profile.get("attempts", 0)) if learning_profile else 0
    required_specialization = str(observation.get("required_specialization", ""))
    critical_patient = observation.get("patient_condition", "").lower() in {"critical", "unstable"}

    # Hard realism rule: never immediately retry the hospital that just rejected the patient.
    if (last_status == "rejected" or is_rerouting_phase) and recent_hospital:
        immediate_retry_block = [item for item in candidates if item["hospital_id"] != recent_hospital]
        if immediate_retry_block:
            candidates = immediate_retry_block
        elif len(candidates) == 1 and candidates[0]["hospital_id"] == recent_hospital:
            fallback_any = [item for item in scored if item["hospital_id"] != recent_hospital]
            if fallback_any:
                candidates = fallback_any

    # In critical non-general cases, prioritize exact specialization when available.
    if critical_patient and required_specialization != "general":
        exact_spec_candidates = [
            item for item in candidates if item["specialization"] == required_specialization
        ]
        if exact_spec_candidates:
            candidates = exact_spec_candidates

    if step_number == 1:
        policy_mode = "safe"
    elif last_status == "rejected":
        policy_mode = "risk-aware"
    else:
        policy_mode = "balanced"

    safe_weight = 1.0
    if policy_mode == "safe":
        safe_weight *= 0.8
        epsilon *= 0.6
        temperature *= 0.8
    elif policy_mode == "risk-aware":
        epsilon *= 1.1
        temperature *= 0.9

    # Within-episode learning from concrete failure reasons.
    if "wrong hospital specialization" in last_reason:
        strict_spec = [
            item
            for item in candidates
            if item["specialization"] == observation.get("required_specialization")
        ]
        if strict_spec:
            candidates = strict_spec
    if "icu unavailable" in last_reason:
        icu_known = [item for item in candidates if item["icu"] == "available"]
        if icu_known:
            candidates = icu_known
    if "specialist" in last_reason:
        strict_spec = [
            item
            for item in candidates
            if item["specialization"] == observation.get("required_specialization")
        ]
        if strict_spec:
            candidates = strict_spec
    if "overloaded" in last_reason:
        non_high_traffic = [item for item in candidates if item["traffic"] != "high"]
        if non_high_traffic:
            candidates = non_high_traffic
    if "delay" in last_reason:
        candidates = sorted(candidates, key=lambda item: item["distance_km"])

    def learned_utility(item: dict) -> float:
        base = float(item.get("policy_score", 0.0))
        if not learning_profile:
            return base
        step_stats = learning_profile.get("step_stats", {}).get(str(step_number), {})
        stats = step_stats.get(item["hospital_id"], {})
        count = int(stats.get("count", 0))
        if count <= 0:
            exploration_bonus = 0.22 * math.sqrt(max(1.0, math.log(attempts + 2.0)))
            return base + exploration_bonus
        avg_reward = float(stats.get("avg_reward", 0.0))
        success_rate = float(stats.get("success_rate", 0.0))
        rejected = int(stats.get("rejected", 0))
        rejection_rate = rejected / max(1, count)
        exploration_bonus = 0.18 * math.sqrt(max(0.0, math.log(attempts + 2.0) / (count + 1.0)))
        # Real-data utility: reward trend + success rate - rejection risk + exploration bonus.
        historical_weight = 0.35
        historical_weight *= 0.6
        historical_bonus = (avg_reward * historical_weight) + (success_rate * 0.30) - (rejection_rate * 0.22)
        if item["hospital_id"] in recent_failed:
            historical_bonus = 0.0
        return base + historical_bonus + exploration_bonus

    def pick_improvement_candidate(route_choice_id: str | None) -> dict | None:
        if not candidates:
            return None
        ranked = sorted(candidates, key=learned_utility, reverse=True)
        if route_choice_id is None:
            return ranked[0]
        for item in ranked:
            if item["hospital_id"] != route_choice_id:
                return item
        return ranked[0]

    def enforce_score_guard(chosen: dict, strategy: str) -> tuple[dict, str]:
        # Absolute next-step guard: never pick the same hospital immediately after a rejection.
        if last_status == "rejected" and previous_action and chosen.get("hospital_id") == previous_action:
            alternatives = [item for item in scored if item["hospital_id"] != previous_action]
            if alternatives:
                rerouted = max(alternatives, key=lambda item: float(item.get("policy_score", 0.0)))
                return rerouted, strategy + " + immediate-retry block"

        # Global guardrail: when a score gap is very large, prefer best option most
        # of the time while preserving some exploration.
        globally_eligible = [
            item
            for item in scored
            if item["hospital_id"] not in recent_failed
            and not (
                (last_status == "rejected" or is_rerouting_phase)
                and recent_hospital
                and item["hospital_id"] == recent_hospital
            )
        ]
        if not globally_eligible:
            globally_eligible = list(scored)

        if globally_eligible:
            best_global = max(globally_eligible, key=lambda item: float(item.get("policy_score", 0.0)))
            chosen_score = float(chosen.get("policy_score", 0.0))
            best_global_score = float(best_global.get("policy_score", 0.0))
            # Cooldown hard guard: never immediately retry the just-failed hospital.
            if (last_status == "rejected" or is_rerouting_phase) and recent_hospital:
                if chosen.get("hospital_id") == recent_hospital:
                    alternatives = [
                        item
                        for item in scored
                        if item["hospital_id"] != recent_hospital and item["hospital_id"] not in recent_failed
                    ]
                    if not alternatives:
                        alternatives = [item for item in scored if item["hospital_id"] != recent_hospital]
                    if alternatives:
                        rerouted = max(alternatives, key=lambda item: float(item.get("policy_score", 0.0)))
                        return rerouted, strategy + " + cooldown reroute"

            if chosen_score < (best_global_score * 0.6):
                return best_global, strategy + " + anti-stupidity guard"
            if (best_global_score - chosen_score) > 0.25 and rng.random() < 0.75:
                return best_global, strategy + " + score-gap guard"

        return chosen, strategy

    # Learning-driven fail guard: avoid hospitals that repeatedly fail at this exact step.
    if learning_profile:
        step_stats = learning_profile.get("step_stats", {}).get(str(step_number), {})
        guard_blocked: set[str] = set()
        for hospital_id, stats in step_stats.items():
            count = int(stats.get("count", 0))
            success_rate = float(stats.get("success_rate", 0.0))
            rejected = int(stats.get("rejected", 0))
            if count >= 2 and success_rate <= 0.0 and rejected >= 2:
                guard_blocked.add(hospital_id)

        guarded_candidates = [item for item in candidates if item["hospital_id"] not in guard_blocked]
        if guarded_candidates:
            candidates = guarded_candidates

    # As attempts increase, reduce randomness and rely on learned utility.
    if attempts >= 3:
        epsilon *= 0.35
        temperature *= 0.70

    # Same seed + same task policy:
    # evaluate route combinations across all steps, not just one-step mutations.
    if learning_profile and policy_mode != "risk-aware":
        best_route = list(learning_profile.get("best_actions", []))
        if step_number - 1 < len(best_route):
            baseline_id = best_route[step_number - 1]
            ranked = sorted(candidates, key=learned_utility, reverse=True)
            baseline_candidate = next((item for item in ranked if item["hospital_id"] == baseline_id), None)
            alternatives = [item for item in ranked if item["hospital_id"] != baseline_id]
            top_candidate = ranked[0] if ranked else None

            if (
                step_number == 1
                and baseline_candidate is not None
                and top_candidate is not None
                and float(baseline_candidate.get("policy_score", 0.0)) < float(top_candidate.get("policy_score", 0.0))
            ):
                baseline_candidate = None

            alternatives = alternatives[: min(3, len(alternatives))]

            if attempts >= 1:
                # Mixed-radix route search: each run selects a step-wise digit.
                # digit 0 => keep baseline for this step, 1/2 => try alternative ranks.
                combo_index = max(0, attempts - 1)
                digit = (combo_index // (3 ** max(0, step_number - 1))) % 3

                if digit == 0 and baseline_candidate is not None:
                    return enforce_score_guard(baseline_candidate, "best-route retain")

                alt_rank = digit - 1
                if alt_rank >= 0 and alt_rank < len(alternatives):
                    return enforce_score_guard(alternatives[alt_rank], f"combination search step-{step_number} alt-{alt_rank + 1}")

                if baseline_candidate is not None:
                    return enforce_score_guard(baseline_candidate, "best-route retain")

    if attempts >= 6:
        ranked = sorted(candidates, key=learned_utility, reverse=True)
        top_pool = ranked[: min(3, len(ranked))]
        return enforce_score_guard(_sample_softmax(top_pool, "policy_score", max(0.08, temperature * 0.85), rng), "learned utility exploit")

    if learning_profile and policy_mode == "safe":
        preferred_route = list(learning_profile.get("best_actions", []))
        if step_number - 1 < len(preferred_route):
            preferred_hospital = preferred_route[step_number - 1]
            preferred_candidate = next((item for item in candidates if item["hospital_id"] == preferred_hospital), None)
            if preferred_candidate is not None:
                profile_score = float(learning_profile.get("best_score", 0.0))
                if (profile_score * safe_weight) >= 0.85 or len(candidates) == 1:
                    return enforce_score_guard(preferred_candidate, "learned best path")

    # If last outcome was partial, force trying a different hospital when possible.
    if last_status == "partial" and previous_action:
        redirected = [item for item in candidates if item["hospital_id"] != previous_action]
        if redirected:
            candidates = redirected
        # After partial treatment, reduce random exploration and favor safer follow-up routing.
        epsilon = min(epsilon, 0.04)
        temperature = min(temperature, 0.24)

    critical = observation.get("patient_condition", "").lower() in {"critical", "unstable"}
    strategy = f"{policy_mode} policy"

    if critical and policy_mode in {"safe", "balanced"}:
        confirmed = [item for item in candidates if item["icu"] == "available"]
        if confirmed:
            candidates = confirmed
            strategy = f"{policy_mode} policy + critical triage"

    if len(candidates) > 1 and rng.random() < 0.15:
        ranked = sorted(candidates, key=learned_utility, reverse=True)
        top_k = ranked[: min(3, len(ranked))]
        return enforce_score_guard(rng.choice(top_k), strategy + " + guided-exploration")

    if len(candidates) > 1:
        # Utility-aware candidate ordering for softmax sampling.
        ranked = sorted(candidates, key=learned_utility, reverse=True)
        chosen = _sample_softmax(ranked, "policy_score", temperature, rng)
        return enforce_score_guard(chosen, strategy)

    return enforce_score_guard(candidates[0], strategy)


def print_options(scored: list[dict]) -> None:
    print(f"Hospital options ({len(scored)} total):")
    for idx, item in enumerate(scored, start=1):
        print(
            f"  [{idx}] {item['hospital_id']} | {item['distance_km']:.1f} km | ICU {item['icu']} | "
            f"traffic {item['traffic']} | specialty {item['specialization']} | score {item['policy_score']:.3f}"
        )


def run_episode(
    env: EmergencyEnv,
    task_id: str,
    seed: int,
    archive: dict | None = None,
    llm_client: object | None = None,
    model_name: str | None = None,
) -> dict:
    observation_model = env.reset(seed=seed, task_id=task_id)
    observation = observation_model.model_dump()
    learning_profile = None
    if archive is not None:
        learning_profile = build_learning_profile(
            archive,
            seed,
            task_id,
            required_specialization=str(observation.get("required_specialization", "")) or None,
        )

    print("\n" + "=" * 72)
    print(f"Scenario: {observation['scenario_name']}")
    print(f"Task: {task_id} | Difficulty: {observation['scenario_difficulty']} | Seed: {seed}")
    print(f"Patient condition: {observation['patient_condition']}")
    print(f"Required specialization: {observation['required_specialization']}")
    print("Objective: admit patient successfully (no fixed deadline window)")
    print("=" * 72)
    emit_structured(
        "START",
        {
            "task_id": task_id,
            "seed": seed,
            "difficulty": observation.get("scenario_difficulty"),
            "scenario": observation.get("scenario_name"),
            "patient_condition": observation.get("patient_condition"),
            "required_specialization": observation.get("required_specialization"),
        },
    )

    if learning_profile:
        print(
            f"Learning memory: best historical score {float(learning_profile.get('best_score', 0.0)):.3f} "
            f"across {int(learning_profile.get('attempts', 0))} attempts"
        )
        if learning_profile.get("best_actions"):
            print(f"Best known route: {' -> '.join(learning_profile['best_actions'])}")

    total_reward = 0.0
    steps = 0
    done = False
    previous_policy_hospital_id: str | None = None
    previous_policy_outcome: str | None = None
    attempt_index = int(learning_profile.get("attempts", 0)) if learning_profile else 0
    # Keep scenario deterministic by seed, but vary policy exploration across retries.
    rng = random.Random(seed + (attempt_index * 7919))
    step_records: list[dict] = []

    while not done:
        steps += 1
        print(f"\nStep {observation['step']} | phase={observation['ambulance_status']}")

        scored = score_hospitals(observation, learning_profile=learning_profile)
        chosen, strategy = choose_hospital(scored, observation, rng, learning_profile=learning_profile)

        # Final policy-level guard: no immediate retry of the same hospital after rejection.
        if previous_policy_outcome == "REJECTED" and previous_policy_hospital_id and chosen["hospital_id"] == previous_policy_hospital_id:
            alternatives = [item for item in scored if item["hospital_id"] != previous_policy_hospital_id]
            if alternatives:
                chosen = max(alternatives, key=lambda item: float(item.get("policy_score", 0.0)))
                strategy = strategy + " + immediate-retry override"

        print_options(scored)
        rationale = llm_rationale(llm_client, model_name or "", observation, chosen, strategy)
        print(f"Decision: {chosen['hospital_id']} ({strategy})")

        step_result = env.step(
            Action(
                step=observation["step"],
                hospital_id=chosen["hospital_id"],
                rationale=rationale,
            )
        )
        next_obs_model = step_result["observation"]
        reward = float(step_result["reward"])
        done = bool(step_result["done"])
        info = step_result.get("info", {}) or {}
        next_observation = next_obs_model.model_dump()
        total_reward += reward

        outcome = info.get("outcome", {})
        status = str(outcome.get("status", "partial")).upper()
        reason = str(outcome.get("reason", "No reason provided"))
        previous_policy_hospital_id = chosen["hospital_id"]
        previous_policy_outcome = status

        print(f"Outcome: {status}")
        print(f"Reason: {reason}")
        print(f"Reward: {reward:.3f}")
        emit_structured(
            "STEP",
            {
                "task_id": task_id,
                "seed": seed,
                "step": observation.get("step"),
                "phase": observation.get("ambulance_status"),
                "hospital_id": chosen["hospital_id"],
                "strategy": strategy,
                "status": status,
                "reward": round(reward, 4),
                "done": done,
            },
        )

        append_trajectory_log(
            {
                "seed": seed,
                "task": task_id,
                "difficulty": observation.get("scenario_difficulty"),
                "step": observation.get("step"),
                "state": {
                    "patient_condition": observation.get("patient_condition"),
                    "remaining_time_minutes": observation.get("remaining_time_minutes"),
                    "failed_hospitals": observation.get("failed_hospitals", []),
                    "visited_hospitals": observation.get("visited_hospitals", []),
                    "ambulance_status": observation.get("ambulance_status"),
                },
                "action": {
                    "hospital_id": chosen["hospital_id"],
                    "policy_score": chosen["policy_score"],
                    "strategy": strategy,
                },
                "outcome": {
                    "status": status,
                    "reason": reason,
                },
                "reward": reward,
            }
        )

        step_records.append(
            {
                "step": observation.get("step"),
                "hospital_id": chosen["hospital_id"],
                "status": status,
                "reason": reason,
                "reward": reward,
                "policy_score": chosen["policy_score"],
            }
        )

        observation = next_observation

    final_state = env.state()
    final_result = final_state.final_outcome or "FAILURE"
    final_score = float(final_state.final_score)

    print("\nFinal result:")
    print(f"  Result: {final_result}")
    print(f"  Total steps: {steps}")
    print(f"  Final score: {final_score:.3f}")
    print(f"  Average reward: {total_reward / max(1, steps):.3f}")
    emit_structured(
        "END",
        {
            "task_id": task_id,
            "seed": seed,
            "result": final_result,
            "success": final_result == "SUCCESS",
            "score": round(final_score, 4),
            "steps": steps,
            "average_reward": round(total_reward / max(1, steps), 4),
        },
    )

    return {
        "success": final_result == "SUCCESS",
        "score": final_score,
        "steps": steps,
        "seed": seed,
        "task_id": task_id,
        "scenario_name": observation.get("scenario_name"),
        "scenario_type": observation.get("scenario_type"),
        "difficulty": observation.get("scenario_difficulty"),
        "required_specialization": observation.get("required_specialization"),
        "actions": [record["hospital_id"] for record in step_records],
        "step_records": step_records,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def update_learning_archive(archive: dict, episode_result: dict) -> None:
    key = profile_key(int(episode_result["seed"]), str(episode_result["task_id"]))
    profiles = archive.setdefault("profiles", {})
    profile = profiles.get(
        key,
        {
            "attempts": 0,
            "best_score": 0.0,
            "best_actions": [],
            "best_steps": 0,
            "step_stats": {},
        },
    )

    profile["attempts"] = int(profile.get("attempts", 0)) + 1
    profile["last_score"] = float(episode_result["score"])
    profile["last_success"] = bool(episode_result["success"])
    profile["last_run_at"] = episode_result["timestamp"]
    profile["last_actions"] = list(episode_result.get("actions", []))
    profile["last_required_specialization"] = episode_result.get("required_specialization")
    profile["last_scenario_type"] = episode_result.get("scenario_type")
    profile["last_scenario_name"] = episode_result.get("scenario_name")

    if float(episode_result["score"]) >= float(profile.get("best_score", 0.0)):
        profile["best_score"] = float(episode_result["score"])
        profile["best_actions"] = list(episode_result.get("actions", []))
        profile["best_steps"] = int(episode_result.get("steps", 0))
        profile["best_success"] = bool(episode_result["success"])
        profile["best_scenario_name"] = episode_result.get("scenario_name")
        profile["best_difficulty"] = episode_result.get("difficulty")
        profile["best_required_specialization"] = episode_result.get("required_specialization")

    step_stats = profile.setdefault("step_stats", {})
    for record in episode_result.get("step_records", []):
        step_key = str(record.get("step"))
        hospital_id = str(record.get("hospital_id"))
        step_bucket = step_stats.setdefault(step_key, {})
        hospital_bucket = step_bucket.setdefault(
            hospital_id,
            {
                "count": 0,
                "success": 0,
                "accepted": 0,
                "partial": 0,
                "rejected": 0,
                "total_reward": 0.0,
                "avg_reward": 0.0,
                "last_status": None,
                "last_reason": None,
            },
        )
        hospital_bucket["count"] += 1
        if record["status"] == "ACCEPTED":
            hospital_bucket["success"] += 1
            hospital_bucket["accepted"] += 1
        elif record["status"] == "PARTIAL":
            hospital_bucket["partial"] += 1
        else:
            hospital_bucket["rejected"] += 1
        hospital_bucket["total_reward"] = float(hospital_bucket["total_reward"]) + float(record["reward"])
        hospital_bucket["avg_reward"] = hospital_bucket["total_reward"] / max(1, hospital_bucket["count"])
        hospital_bucket["last_status"] = record["status"]
        hospital_bucket["last_reason"] = record["reason"]
        hospital_bucket["success_rate"] = hospital_bucket["accepted"] / max(1, hospital_bucket["count"])

    profiles[key] = profile
    episodes = archive.setdefault("episodes", [])
    episodes.append(
        {
            "seed": episode_result["seed"],
            "task_id": episode_result["task_id"],
            "difficulty": episode_result["difficulty"],
            "required_specialization": episode_result.get("required_specialization"),
            "scenario_name": episode_result["scenario_name"],
            "score": episode_result["score"],
            "success": episode_result["success"],
            "actions": episode_result.get("actions", []),
            "timestamp": episode_result["timestamp"],
        }
    )
    archive["episodes"] = episodes[-500:]


def print_training_summary(results: list[dict]) -> None:
    if not results:
        return
    scores = [float(item["score"]) for item in results]
    successes = sum(1 for item in results if item["success"])
    split = max(1, len(scores) // 2)
    early_scores = scores[:split]
    late_scores = scores[split:]
    if not late_scores:
        late_scores = scores[-split:]
    early_avg = sum(early_scores) / len(early_scores)
    late_avg = sum(late_scores) / len(late_scores)
    delta = late_avg - early_avg

    print("\nTraining summary:")
    print(f"  Episodes: {len(results)}")
    print(f"  Success rate: {successes / len(results):.1%}")
    print(f"  Average score: {sum(scores) / len(scores):.3f}")
    print(f"  Early avg score ({len(early_scores)} eps): {early_avg:.3f}")
    print(f"  Late avg score ({len(late_scores)} eps): {late_avg:.3f}")
    print(f"  Trend delta (late-early): {delta:+.3f}")


def main() -> None:
    args = parse_args()
    llm_client, model_name = require_llm_config()
    seed = ask_seed_if_missing(args.seed)
    print(f"Using seed: {seed}")
    if args.mode == "full":
        tasks = TASK_ORDER
    else:
        chosen_task = args.task
        if chosen_task is None:
            chosen_level = ask_level_if_missing(args.level)
            chosen_task = LEVEL_TO_TASK[chosen_level]
        tasks = [chosen_task]

    env = EmergencyEnv(memory_file=args.memory_file)
    archive = load_learning_archive()

    results = []
    run_count = args.train_episodes if args.train_episodes > 0 else args.episodes
    training_mode = args.train_episodes > 0

    for episode in range(run_count):
        for idx, task_id in enumerate(tasks):
            if training_mode:
                if args.train_same_seed:
                    task_seed = seed
                else:
                    task_seed = seed + (episode * 100) + idx
            else:
                task_seed = seed + (episode * 100) + idx

            label = f"Training Episode {episode + 1}" if training_mode else f"Episode {episode + 1}"
            print(f"\n=== {label} | {task_id} | seed={task_seed} ===")
            episode_result = run_episode(
                env,
                task_id,
                task_seed,
                archive=archive,
                llm_client=llm_client,
                model_name=model_name,
            )
            results.append(episode_result)
            update_learning_archive(archive, episode_result)

    save_learning_archive(archive)

    if training_mode:
        print_training_summary(results)
        return

    if results:
        print("\nBatch summary:")
        if len(results) == 1:
            episode_result = "SUCCESS" if results[0]["success"] else "FAILURE"
            print(f"  Episode outcome: {episode_result}")
            print(f"  Episode score: {results[0]['score']:.3f}")
            print(f"  Episode steps: {results[0]['steps']}")
            print("  Note: run 30-50 episodes to estimate difficulty success rate.")
        else:
            print(f"  Success rate: {sum(1 for item in results if item['success']) / len(results):.1%}")
            print(f"  Average score: {sum(item['score'] for item in results) / len(results):.3f}")
            print(f"  Average steps: {sum(item['steps'] for item in results) / len(results):.1f}")


if __name__ == "__main__":
    main()
