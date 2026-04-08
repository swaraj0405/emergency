# ACDE OpenEnv Submission

This repository contains the Adaptive Crisis Decision Environment (ACDE) project and submission assets.

## Main Submission Package

The deployable OpenEnv project is in:
- `my_env/`

Key files inside `my_env/`:
- `openenv.yaml` - OpenEnv contract
- `inference.py` - baseline inference runner
- `app/server/app.py` - FastAPI endpoints
- `README.md` - detailed environment documentation

## Deployed Hugging Face Space

- Space page: https://huggingface.co/spaces/paramjitbaral/acde-openenv
- Live endpoint: https://paramjitbaral-acde-openenv.hf.space

## Quick Local Run

From the repository root:

```powershell
cd my_env
python inference.py --mode full --episodes 3 --seed 555
```

Health check for deployed API:

```text
GET https://paramjitbaral-acde-openenv.hf.space/health
```
