#!/usr/bin/env python3
"""Pre-submission validation checks for ACDE OpenEnv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib import error, request

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


TASK_ORDER = ["acde_easy", "acde_medium", "acde_hard"]
EXPECTED_DIFFICULTIES = ["easy", "medium", "hard"]
STRICT_MIN = 0.0
STRICT_MAX = 1.0


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def is_strict_score(value: float) -> bool:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return False
    return STRICT_MIN < score < STRICT_MAX


def post_json(url: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="ignore")
        try:
            payload_json = json.loads(text)
            raise RuntimeError(payload_json.get("detail", text) or str(exc)) from exc
        except json.JSONDecodeError:
            raise RuntimeError(text or str(exc)) from exc


def get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    req = request.Request(url, method="GET")
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def find_openenv_yaml(repo_root: Path) -> Path | None:
    candidates = [
        repo_root / "openenv.yaml",
        repo_root / "my_env" / "openenv.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def validate_yaml(path: Path) -> bool:
    if yaml is None:
        fail("PyYAML is not installed. Install pyyaml to run YAML checks.")
        return False

    if not path.exists():
        fail(f"Missing metadata file: {path}")
        return False

    with path.open("r", encoding="utf-8") as fp:
        spec = yaml.safe_load(fp)

    if not isinstance(spec, dict):
        fail("openenv.yaml is not a valid mapping object")
        return False

    required_root = ["name", "version", "runtime", "endpoints", "contracts"]
    missing = [key for key in required_root if key not in spec]
    if missing:
        fail(f"openenv.yaml missing keys: {missing}")
        return False

    endpoints = spec.get("endpoints", {})
    required_endpoints = {
        "health": ("GET", "/health"),
        "reset": ("POST", "/reset"),
        "step": ("POST", "/step"),
        "state": ("GET", "/state"),
    }
    for name, (method, route) in required_endpoints.items():
        entry = endpoints.get(name)
        if not isinstance(entry, dict):
            fail(f"openenv.yaml endpoints missing entry: {name}")
            return False
        if str(entry.get("method", "")).upper() != method:
            fail(f"openenv.yaml endpoint {name} method mismatch")
            return False
        if str(entry.get("path", "")) != route:
            fail(f"openenv.yaml endpoint {name} path mismatch")
            return False

    contracts = spec.get("contracts", {})
    reward_range = contracts.get("reward_range")
    if not (isinstance(reward_range, list) and len(reward_range) == 2):
        fail("openenv.yaml contracts.reward_range is missing or invalid")
        return False

    try:
        low = float(reward_range[0])
        high = float(reward_range[1])
    except (TypeError, ValueError):
        fail("openenv.yaml contracts.reward_range must be numeric")
        return False

    if not (0.0 < low < high < 1.0):
        fail(f"openenv.yaml reward_range must be strictly inside (0,1), got [{low}, {high}]")
        return False

    ok(f"openenv.yaml checks passed: {path}")
    return True


def choose_action(observation: dict[str, Any]) -> dict[str, Any]:
    hospitals = list(observation.get("hospitals", []))
    if not hospitals:
        raise RuntimeError("Observation has no hospitals to choose from.")

    required_specialization = str(observation.get("required_specialization", "general"))
    traffic_rank = {"low": 0, "medium": 1, "high": 2}

    ranked = sorted(
        hospitals,
        key=lambda h: (
            -int(str(h.get("specialization", "")) == required_specialization),
            -int(
                str(h.get("specialization", "")) == required_specialization
                or str(h.get("specialization", "")) == "general"
                or required_specialization == "general"
            ),
            -int(str(h.get("icu", "unknown")) == "available"),
            float(h.get("distance_km", 999.0)),
            traffic_rank.get(str(h.get("traffic", "medium")), 1),
            str(h.get("hospital_id", "")),
        ),
    )

    chosen = ranked[0]
    return {
        "step": int(observation.get("step", 1)),
        "hospital_id": str(chosen["hospital_id"]),
        "rationale": "pre-submit deterministic validator policy",
    }


def validate_local_environment(repo_root: Path, seed: int) -> bool:
    success = True

    my_env_dir = repo_root / "my_env"
    if str(my_env_dir) not in sys.path:
        sys.path.insert(0, str(my_env_dir))

    try:
        from app.environment.core import EmergencyEnv, TASKS
        from app.models.action import Action
    except Exception as exc:
        fail(f"Local import check failed: {exc}")
        return False

    missing_tasks = [task_id for task_id in TASK_ORDER if task_id not in TASKS]
    if missing_tasks:
        fail(f"Local TASKS missing required tasks: {missing_tasks}")
        return False

    progression = [str(TASKS[task_id].get("difficulty")) for task_id in TASK_ORDER]
    if progression != EXPECTED_DIFFICULTIES:
        fail(f"Difficulty progression mismatch: {progression}")
        success = False
    else:
        ok("Task difficulty progression is easy -> medium -> hard")

    tmp_dir = repo_root / ".validator_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for index, task_id in enumerate(TASK_ORDER):
        task_seed = seed + index
        memory_a = tmp_dir / f"{task_id}_a_memory.json"
        memory_b = tmp_dir / f"{task_id}_b_memory.json"
        memory_rollout = tmp_dir / f"{task_id}_rollout_memory.json"
        memory_a.write_text("{}", encoding="utf-8")
        memory_b.write_text("{}", encoding="utf-8")
        memory_rollout.write_text("{}", encoding="utf-8")

        # Determinism on reset with same seed and task.
        env_a = EmergencyEnv(memory_file=str(memory_a))
        env_b = EmergencyEnv(memory_file=str(memory_b))
        obs_a = env_a.reset(seed=task_seed, task_id=task_id)
        obs_b = env_b.reset(seed=task_seed, task_id=task_id)
        if obs_a.model_dump() != obs_b.model_dump():
            fail(f"Reset determinism failed for {task_id}")
            success = False
        else:
            ok(f"Reset determinism passed for {task_id}")

        # Deterministic rollout checks for reward bounds and final score bounds.
        env = EmergencyEnv(memory_file=str(memory_rollout))
        obs = env.reset(seed=task_seed, task_id=task_id)
        done = False
        last_step_payload: dict[str, Any] | None = None
        max_steps = int(obs.max_steps)

        for _ in range(max_steps + 2):
            action_payload = choose_action(obs.model_dump())
            result = env.step(
                Action(
                    step=int(action_payload["step"]),
                    hospital_id=str(action_payload["hospital_id"]),
                    rationale=str(action_payload["rationale"]),
                )
            )
            reward = float(result.get("reward", -1.0))
            if not is_strict_score(reward):
                fail(f"Reward out of strict (0,1) for {task_id}: {reward}")
                success = False
                break

            last_step_payload = result
            done = bool(result.get("done"))
            obs = result["observation"]
            if done:
                break

        if not done:
            fail(f"Episode did not terminate within expected step budget for {task_id}")
            success = False

        final_score = float(env.state().final_score)
        if is_strict_score(final_score):
            ok(f"Final state score strict bounds passed for {task_id}: score={final_score:.3f}")
        else:
            fail(f"Final state score not in strict (0,1) for {task_id}: score={final_score}")
            success = False

        info = (last_step_payload or {}).get("info", {}) if isinstance(last_step_payload, dict) else {}
        grader = info.get("grader", {}) if isinstance(info, dict) else {}
        if grader:
            grader_score = float(grader.get("score", -1.0))
            if is_strict_score(grader_score):
                ok(f"Grader strict bounds passed for {task_id}: score={grader_score:.3f}")
            else:
                fail(f"Grader score not strictly in (0,1) for {task_id}: score={grader_score}")
                success = False

    # Best-effort cleanup of temporary files.
    for path in tmp_dir.glob("*_memory.json"):
        try:
            path.unlink()
        except OSError:
            pass
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    return success


def validate_live_api(env_url: str, seed: int) -> bool:
    success = True

    try:
        health = get_json(f"{env_url}/health")
        if str(health.get("status", "")).lower() != "ok":
            fail(f"Health endpoint responded but status is unexpected: {health}")
            return False
        ok("Health endpoint reachable")
    except Exception as exc:
        fail(f"Health endpoint failed: {exc}")
        return False

    for index, task_id in enumerate(TASK_ORDER):
        task_seed = seed + index
        try:
            reset = post_json(f"{env_url}/reset", {"task_id": task_id, "seed": task_seed})
            observation = reset.get("observation", {})
            if not observation:
                raise RuntimeError("reset returned empty observation")

            max_steps = int(observation.get("max_steps", 4))
            done = False
            last_step: dict[str, Any] | None = None

            for _ in range(max_steps + 2):
                action = choose_action(observation)
                step = post_json(f"{env_url}/step", action)
                reward = float(step.get("reward", -1.0))
                if not is_strict_score(reward):
                    raise RuntimeError(f"step reward not strictly in (0,1): {reward}")

                last_step = step
                done = bool(step.get("done"))
                if done:
                    break
                observation = step.get("observation", {})

            if not done:
                raise RuntimeError("episode did not terminate within expected steps")

            score_source = "info.grader.score"
            raw_score: float | None = None
            info = (last_step or {}).get("info", {}) if isinstance(last_step, dict) else {}
            grader = info.get("grader", {}) if isinstance(info, dict) else {}
            if grader and ("score" in grader):
                raw_score = float(grader["score"])
            else:
                state_payload = get_json(f"{env_url}/state")
                if state_payload.get("task_id") != task_id:
                    raise RuntimeError(
                        f"/state task_id mismatch: expected {task_id}, got {state_payload.get('task_id')}"
                    )
                raw_score = float(state_payload.get("final_score", -1.0))
                score_source = "state.final_score"

            if not is_strict_score(raw_score):
                raise RuntimeError(f"task score not strictly in (0,1): {raw_score}")

            ok(f"Live flow passed for {task_id}: score={raw_score:.3f} via {score_source}")
        except Exception as exc:
            fail(f"Live flow failed for {task_id}: {exc}")
            success = False

    return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ACDE OpenEnv submission readiness")
    parser.add_argument(
        "--env-url",
        default="http://127.0.0.1:7860",
        help="Running API server URL",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260409,
        help="Base seed for deterministic checks",
    )
    parser.add_argument(
        "--openenv-path",
        default=None,
        help="Optional explicit path to openenv.yaml",
    )
    parser.add_argument(
        "--check-yaml-only",
        action="store_true",
        help="Only validate openenv.yaml",
    )
    parser.add_argument(
        "--skip-local",
        action="store_true",
        help="Skip local in-process environment checks",
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Skip live API checks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    env_url = args.env_url.rstrip("/")

    openenv_path = Path(args.openenv_path).resolve() if args.openenv_path else find_openenv_yaml(repo_root)
    if openenv_path is None:
        fail("Could not locate openenv.yaml (checked ./openenv.yaml and ./my_env/openenv.yaml)")
        return 1

    checks: list[bool] = []
    checks.append(validate_yaml(openenv_path))

    if not args.check_yaml_only:
        if not args.skip_local:
            checks.append(validate_local_environment(repo_root, args.seed))
        if not args.skip_live:
            checks.append(validate_live_api(env_url, args.seed))

    if all(checks):
        print("\nAll validation checks passed.")
        return 0

    print("\nValidation failed. Review [FAIL] lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
