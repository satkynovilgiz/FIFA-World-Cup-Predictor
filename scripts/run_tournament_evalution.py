import pandas as pd

from core.models import MODEL_BUILDERS_MAP
from core.constants import TOURNAMENT_EVAL_PATH
from core.tuning import TournamentEvaluator

if __name__ == "__main__":
  result = []
  for model_id, model_builder in MODEL_BUILDERS_MAP.items():
    result.extend(
      TournamentEvaluator(model_id, model_builder).run())
    df = pd.DataFrame(result).sort_values("tournament_loss")
    df.to_csv(TOURNAMENT_EVAL_PATH, index=False)
    print(f"Result saved at {TOURNAMENT_EVAL_PATH}")

  print("All evaluations saved!")
