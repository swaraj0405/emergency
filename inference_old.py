#!/usr/bin/env python3
"""ACDE inference runner with simple interactive flow and detailed explanations."""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib import error, request

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_NAME = os.getenv("ENV_NAME", "acde-openenv")
TASK_ORDER = ["acde_easy", "acde_medium", "acde_hard"]
TRAFFIC_FACTOR = {"low": 1.0, "medium": 0.6, "high": 0.3}
BASE_SPEED_KMH = 60.0
SCORE_FLOOR = 0.001
SCORE_CEIL = 0.999


def clamp_task_score(value: float) -> float:
    """Clamp score-like values to the strict open interval (0, 1)."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = SCORE_FLOOR
    return max(SCORE_FLOOR, min(SCORE_CEIL, score))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACDE inference runner")
    parser.add_argument("--mode", choices=["single", "full"], default="single")
    parser.add_argument("--task", choices=TASK_ORDER, default="acde_medium")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--strict", action="store_true", help="Use strict [START]/[STEP]/[END] output")
    parser.add_argument(
        "--memory-file",
        default=str(Path(__file__).resolve().parent / "my_env" / "data" / "learning_memory.json"),
    )
    return parser.parse_args()


def ask_seed_if_missing(seed: int | None) -> int:
    if seed is not None:
        return seed
    raw = input("Enter seed number (example 555): ").strip()
    try:
        return int(raw)
    except ValueError:
        print("Invalid seed. Using default seed=202601")
        return 202601


def post_json(url: str, payload: dict, timeout: int = 20) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
            return json.loads(text)
    except error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="ignore")
        try:
            data = json.loads(text)
            raise RuntimeError(data.get("detail", text) or str(exc)) from exc
        except json.JSONDecodeError:
            raise RuntimeError(text or str(exc)) from exc
    except error.URLError as exc:
        raise ConnectionError(str(exc)) from exc


def load_memory(memory_file: str) -> dict:
    path = Path(memory_file)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def memory_score_for_hospital(hospital_id: str, memory: dict) -> float:
    entry = memory.get(hospital_id)
    if not entry:
        return 0.5

    success = int(entry.get("success", 0))
    fail = int(entry.get("fail", 0))
    avg = float(entry.get("avg", 0.0))
    total = success + fail
    if total == 0:
        return 0.5

    success_rate = success / total
    fail_bias = max(0.0, (fail - success) / total)
    raw = (0.7 * avg) + (0.3 * success_rate) - (0.4 * fail_bias)
    return max(0.0, min(1.0, raw))


def score_options(observation: dict, memory_file: str) -> list[dict]:
    memory = load_memory(memory_file)
    scored: list[dict] = []
    failed_hospitals = set(observation.get("failed_hospitals", []))
    visited_hospitals = set(observation.get("visited_hospitals", []))
    remaining_time = float(observation.get("remaining_time_minutes", observation["critical_time_limit_minutes"]))
    initial_limit = float(observation.get("initial_critical_time_limit_minutes", observation["critical_time_limit_minutes"]))
    urgency = 1.0 - max(0.0, min(1.0, remaining_time / max(initial_limit, 1e-6)))

    for hospital in observation["hospitals"]:
        icu_score = 1.0 if hospital["icu"] == "available" else 0.55
        distance_score = max(0.0, min(1.0, 1.0 - (hospital["distance_km"] / 20.0)))
        traffic_score = TRAFFIC_FACTOR[hospital["traffic"]]
        memory_score = memory_score_for_hospital(hospital["hospital_id"], memory)
        specialization_match = (
            hospital["specialization"] == observation["required_specialization"]
            or hospital["specialization"] == "general"
            or observation["required_specialization"] == "general"
        )
        base_score = (
            (icu_score * 0.4)
            + (distance_score * 0.3)
            + (traffic_score * 0.2)
            + (memory_score * 0.3)
        ) / 1.2

        # Journey continuity penalties and urgency boost.
        failed_penalty = 0.24 if hospital["hospital_id"] in failed_hospitals else 0.0
        revisit_penalty = 0.08 if hospital["hospital_id"] in visited_hospitals else 0.0
        urgency_boost = urgency * ((distance_score * 0.6) + (traffic_score * 0.4)) * 0.32
        reroute_explore_bonus = 0.1 if (failed_hospitals and hospital["hospital_id"] not in visited_hospitals) else 0.0
        capability_bonus = 0.06 if specialization_match else 0.0
        decision_score = max(
            0.0,
            min(
                1.0,
                base_score
                + urgency_boost
                + reroute_explore_bonus
                + capability_bonus
                - failed_penalty
                - revisit_penalty,
            ),
        )

        speed_kmh = BASE_SPEED_KMH * TRAFFIC_FACTOR[hospital["traffic"]]
        travel_time = (hospital["distance_km"] / speed_kmh) * 60.0

        scored.append(
            {
                "hospital_id": hospital["hospital_id"],
                "icu": hospital["icu"],
                "distance_km": hospital["distance_km"],
                "traffic": hospital["traffic"],
                "specialization": hospital["specialization"],
                "icu_score": icu_score,
                "distance_score": distance_score,
                "traffic_score": traffic_score,
                "memory_score": memory_score,
                "decision_score": max(0.0, min(1.0, decision_score)),
                "speed_kmh": speed_kmh,
                "travel_time": travel_time,
                "failed_penalty": failed_penalty,
                "revisit_penalty": revisit_penalty,
                "urgency": urgency,
                "specialization_match": specialization_match,
            }
        )

    scored.sort(key=lambda x: x["decision_score"], reverse=True)
    return scored


def top_memory_preference(memory_file: str) -> str:
    memory = load_memory(memory_file)
    if not memory:
        return "none"
    scored = sorted(
        memory.items(),
        key=lambda kv: memory_score_for_hospital(kv[0], memory),
        reverse=True,
    )
    hid, row = scored[0]
    return f"{hid} (avg={float(row.get('avg', 0.0)):.3f})"


def extract_bool(explanation: list[str], prefix: str) -> bool | None:
    for line in explanation:
        if line.startswith(prefix):
            value = line.split("=", 1)[1].strip().lower()
            if value == "true":
                return True
            if value == "false":
                return False
    return None


def extract_state_changes(explanation: list[str]) -> list[str]:
    updates: list[str] = []
    for line in explanation:
        if ": traffic " in line and "->" in line:
            # Example: "H1: traffic high->low, ICU actual True->False"
            hospital_id, details = line.split(": ", 1)
            for part in [p.strip() for p in details.split(",")]:
                if part.startswith("traffic "):
                    updates.append(f"{hospital_id} traffic {part.replace('traffic ', '')}")
                elif part.startswith("ICU actual "):
                    updates.append(f"{hospital_id} ICU actual {part.replace('ICU actual ', '')}")
        elif line.startswith("Patient condition worsened:"):
            updates.append(line.replace("Patient condition worsened: ", "patient "))
    return updates


def short_reason(chosen: dict, second: dict | None) -> str:
    reasons: list[str] = []
    if chosen["failed_penalty"] > 0:
        reasons.append("reselected failed site only due to strong urgency")
    if chosen["icu"] == "available":
        reasons.append("ICU looked available")
    if second and chosen["travel_time"] <= second["travel_time"]:
        reasons.append("faster arrival")
    if chosen["memory_score"] >= 0.55:
        reasons.append("strong past reliability")
    if chosen["urgency"] >= 0.5:
        reasons.append("low remaining time priority")
    if not reasons:
        reasons.append("best overall weighted decision")
    return "; ".join(reasons[:2])


def option_lines(observation: dict) -> list[str]:
    lines: list[str] = []
    for h in observation["hospitals"]:
        lines.append(
            f"  - {h['hospital_id']}: {h['distance_km']:.1f}km | ICU {h['icu']} | traffic {h['traffic']}"
        )
    return lines


def step_result_label(step_payload: dict) -> str:
    info = step_payload.get("info", {}) or {}
    breakdown = ((info.get("reward_model") or {}).get("breakdown") or {})
    survival = float(breakdown.get("survival_component", 0.0))
    if survival >= 0.65:
        return "SUCCESS"
    if survival >= 0.4:
        return "PARTIAL"
    return "FAILURE"


def summarize_task(task_id: str, step_notes: list[str], final_score: float, passed: bool) -> None:
    mistakes = [n for n in step_notes if "FAILURE" in n]
    adapted = "yes" if len(step_notes) >= 2 and len(set(step_notes[-2:])) > 1 else "partial"
    print("\nScenario Summary")
    print(f"  1) Steps completed: {len(step_notes)}")
    print(f"  2) Final score: {final_score:.3f} ({'SUCCESS' if passed else 'FAILURE'})")
    print(f"  3) Mistakes: {len(mistakes)} failure step(s)")
    print(f"  4) Adaptation: {adapted}")


def run_task_strict(task_id: str, seed: int, memory_file: str) -> None:
    def b(v: bool) -> str:
        return "true" if v else "false"

    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")
    try:
        reset_payload = post_json(f"{ENV_BASE_URL}/reset", {"seed": seed, "task_id": task_id}, timeout=20)
    except Exception as exc:
        print(
            f"[END] success=false steps=0 score={clamp_task_score(0.0):.3f} rewards= error={str(exc)!r}"
        )
        return

    observation = reset_payload["observation"]
    rewards: list[float] = []
    steps = 0
    done = False
    step_payload: dict = {"info": {}}

    while not done and steps < int(observation.get("max_steps", 3)):
        steps += 1
        scored = score_options(observation, memory_file)
        chosen = scored[0]
        hid = chosen["hospital_id"]

        try:
            step_payload = post_json(
                f"{ENV_BASE_URL}/step",
                {
                    "step": observation["step"],
                    "hospital_id": hid,
                    "rationale": "weighted decision with memory influence",
                },
                timeout=20,
            )
            reward = float(step_payload["reward"])
            done = bool(step_payload["done"])
            rewards.append(reward)
            print(
                f"[STEP] step={steps} action=route('{hid}') reward={reward:.2f} done={b(done)} error=null"
            )
            observation = step_payload["observation"]
        except Exception as exc:
            print(
                f"[STEP] step={steps} action=route('{hid}') "
                f"reward={clamp_task_score(0.0):.3f} done=true error={str(exc)!r}"
            )
            done = True

    grader = ((step_payload.get("info") or {}).get("grader") or {})
    fallback_score = (sum(rewards) / len(rewards)) if rewards else SCORE_FLOOR
    score = clamp_task_score(grader.get("score", fallback_score))
    success = bool(grader.get("passed", score >= 0.6))
    rewards_csv = ",".join(f"{clamp_task_score(r):.3f}" for r in rewards)
    print(f"[END] success={b(success)} steps={steps} score={score:.3f} rewards={rewards_csv}")


def run_task_detailed(task_id: str, seed: int, memory_file: str) -> None:
    print("\n" + "-" * 56)
    print(f"Scenario {task_id} | seed={seed}")
    print(f"Memory preference: {top_memory_preference(memory_file)}")

    try:
        reset_payload = post_json(
            f"{ENV_BASE_URL}/reset",
            {"seed": seed, "task_id": task_id},
            timeout=20,
        )
    except Exception as exc:
        print(f"Failed to reset environment: {exc}")
        return

    observation = reset_payload["observation"]
    print(
        f"Context: {observation['scenario_type']} | {observation['scenario_name']} | "
        f"need={observation['required_specialization']} | limit={observation['critical_time_limit_minutes']:.2f}m"
    )
    print(
        f"Journey: status={observation.get('ambulance_status', 'en_route')} | "
        f"location={observation.get('current_location_context', 'incident_site')}"
    )

    done = False
    step_no = 0
    step_notes: list[str] = []
    final_score = SCORE_FLOOR
    final_pass = False
    while not done and step_no < int(observation.get("max_steps", 3)):
        step_no += 1
        print(f"\n[Step {step_no}]")
        if observation.get("ambulance_status") == "rerouting":
            print("Phase: REROUTING")
            if observation.get("rerouting_reason"):
                print(f"Reroute reason: {observation['rerouting_reason']}")
        else:
            print("Phase: INITIAL ROUTE")
        print(
            f"Tried: {observation.get('visited_hospitals', [])} | Failed: {observation.get('failed_hospitals', [])}"
        )
        print("Options:")
        for line in option_lines(observation):
            print(line)

        scored = score_options(observation, memory_file)
        chosen = scored[0]
        chosen_hospital_id = chosen["hospital_id"]
        second = scored[1] if len(scored) > 1 else None

        print(f"Decision: {chosen_hospital_id}")
        print(f"Reason: {short_reason(chosen, second)}")
        print(f"Time: {chosen['travel_time']:.2f}m vs remaining {observation.get('remaining_time_minutes', 0.0):.2f}m")

        try:
            step_payload = post_json(
                f"{ENV_BASE_URL}/step",
                {
                    "step": observation["step"],
                    "hospital_id": chosen_hospital_id,
                    "rationale": "weighted decision with memory influence",
                },
                timeout=20,
            )
        except Exception as exc:
            print(f"Step failed: {exc}")
            return

        result = step_result_label(step_payload)
        reward = float(step_payload["reward"])
        print(f"Result: {result} (reward={reward:.3f})")

        changes = extract_state_changes(step_payload.get("observation", {}).get("explanation", []))
        if changes:
            print("Changes:")
            for change in changes[:4]:
                print(f"  - {change}")

        mem = load_memory(memory_file).get(chosen_hospital_id, {})
        if mem:
            print(
                f"Memory: {chosen_hospital_id} avg={float(mem.get('avg', 0.0)):.3f} "
                f"(s={int(mem.get('success', 0))}, f={int(mem.get('fail', 0))})"
            )

        done = bool(step_payload["done"])
        step_notes.append(f"step={step_no}:{result}")

        if done:
            grader = ((step_payload.get("info") or {}).get("grader") or {})
            final_score = clamp_task_score(grader.get("score", reward))
            final_pass = bool(grader.get("passed", final_score >= 0.6))

        observation = step_payload["observation"]

    summarize_task(task_id, step_notes, clamp_task_score(final_score), final_pass)


def main() -> None:
    args = parse_args()
    seed = ask_seed_if_missing(args.seed)
    tasks = [args.task] if args.mode == "single" else TASK_ORDER

    print(f"Running ACDE | mode={args.mode} | seed={seed}")

    for ep in range(args.episodes):
        for i, task_id in enumerate(tasks):
            task_seed = seed + (ep * 100) + i
            if args.strict:
                run_task_strict(task_id=task_id, seed=task_seed, memory_file=args.memory_file)
            else:
                run_task_detailed(task_id=task_id, seed=task_seed, memory_file=args.memory_file)


if __name__ == "__main__":
    main()
