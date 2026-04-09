from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.environment.core import ACDEEnvironment
from app.models.action import Action
from app.models.observation import Observation
from app.models.reward import StepInfo, GraderResult, RewardModel, RewardBreakdown
from app.models.state import EnvState

ROOT = Path(__file__).resolve().parents[2]
MEMORY_FILE = ROOT / "data" / "learning_memory.json"

app = FastAPI(title="Adaptive Crisis Decision Environment", version="1.0.0")
env = ACDEEnvironment(memory_file=str(MEMORY_FILE))
MIN_REWARD = 0.001


class ResetRequest(BaseModel):
    seed: int | None = None
    task_id: str | None = None


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: StepInfo


@app.get("/")
def root() -> dict:
    return {
        "message": "ACDE API is running",
        "where_to_see_output": "Run 'python inference.py' in terminal. Output is printed in terminal, not browser.",
        "quick_check": "/health",
        "api_docs": "/docs",
    }


@app.get("/web")
def web() -> dict:
    # Some HF views probe /web; return the same healthful payload as root.
    return root()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reset", response_model=StepResponse)
def reset(payload: ResetRequest | None = None) -> StepResponse:
    seed = payload.seed if payload else None
    task_id = payload.task_id if payload else None
    obs = env.reset(seed=seed, task_id=task_id)
    info = env.last_info if env.last_info else StepInfo(task_id="acde_medium", difficulty="medium", objective="", progress_score=MIN_REWARD, reward_model=RewardModel(value=MIN_REWARD, breakdown=RewardBreakdown(survival_component=MIN_REWARD, time_efficiency_component=MIN_REWARD, specialization_component=MIN_REWARD, delay_penalty=MIN_REWARD)), grader=GraderResult(task_id="acde_medium", difficulty="medium", objective="", score=MIN_REWARD, passed=False, criteria={}), last_action_error=None, outcome=None)
    return StepResponse(observation=obs, reward=MIN_REWARD, done=False, info=info)


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    try:
        result = env.step(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StepResponse(
        observation=result["observation"],
        reward=float(result["reward"]),
        done=bool(result["done"]),
        info=result.get("info", {}),
    )


@app.get("/state", response_model=EnvState)
def state() -> EnvState:
    return env.state()
