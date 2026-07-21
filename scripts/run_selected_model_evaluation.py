import pandas as pd

from core.constants import SELECTED_MODEL_RESULT
from core.evaluation import tournament_score
from core.models import load_best_config

if __name__ == "__main__":
  best_model_res = load_best_config()
  iters = 20_000
  score = tournament_score(
    iters=iters,
    model_factory=best_model_res.model_factory,
    feats=best_model_res.feats)
  result = {
    "model_id": best_model_res.model_id,
    "model_name": best_model_res.model_name,
    "iters": iters,
    **score,
  }
  pd.DataFrame([result]).to_csv(SELECTED_MODEL_RESULT, index=False)

  print(f"Result saved at {SELECTED_MODEL_RESULT}")
