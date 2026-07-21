import sys

from core.constants import GRID_SEARCH_RESULT_PATH
from core.tuning import GridSearch, GRID_MAP

if __name__ == "__main__":
  if len(sys.argv) < 2:
    raise ValueError(f"Model name is required as argument")
  model_id = sys.argv[1]
  builder, grid = GRID_MAP[model_id]
  result = GridSearch(builder, grid).run()
  result_path = f"{GRID_SEARCH_RESULT_PATH}/{model_id}.csv"
  result.to_csv(result_path, index=False)
  print(f"Result saved at {result_path}")
