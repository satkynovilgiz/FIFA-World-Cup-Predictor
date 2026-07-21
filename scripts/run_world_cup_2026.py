import sys
from pathlib import Path

import pandas as pd

from core.constants import (WC_2026_RESULT_CHAMPION_FILE, WC_2026_RESULT_DIR, WC_2026_RESULT_FINAL_FILE,
                            WC_2026_RESULT_SEMI_FILE, WC_2026_RESULT_QUARTER_FILE, WC_2026_RESULT_16_FILE,
                            WC_2026_RESULT_32_FILE)
from core.models import load_best_config
from core.monte_carlo import YearProbCounter, run_wc_2026_monte_carlo
from core.world_cup_sim import WorldCupSimResult

_YEAR = 2026

if len(sys.argv) < 3:
  raise ValueError(f"Num of monte carlo iterations and random state are required as argument")
iters = int(sys.argv[1])
rand_state = int(sys.argv[2])


class WC2026Collector:
  def __init__(self):
    self.champion = YearProbCounter(_YEAR)
    self.final = YearProbCounter(_YEAR)
    self.semi = YearProbCounter(_YEAR)
    self.quarter = YearProbCounter(_YEAR)
    self.last_16 = YearProbCounter(_YEAR)
    self.last_32 = YearProbCounter(_YEAR)

  def on_complete_iter(self, res: WorldCupSimResult):
    self.champion.count(_YEAR, res.champion)
    self.final.count(_YEAR, [res.champion, res.runner_up])
    self.semi.count(_YEAR, [res.champion, res.runner_up, res.third, res.fourth])
    self.quarter.count(_YEAR, res.quarter_finalists)
    self.last_16.count(_YEAR, res.last_16)
    self.last_32.count(_YEAR, res.last_32)


def create_run_dir() -> Path:
  run_dir = Path(
    WC_2026_RESULT_DIR,
    f"{iters}-iters__seed-{rand_state}")

  run_dir.mkdir(parents=True, exist_ok=False)

  return run_dir


def generate_results(counter: YearProbCounter, target_path: Path):
  probs = counter.calculate_probs(iters)
  df = pd.DataFrame(list(probs[_YEAR].items()), columns=["team", "prob"])
  df.to_csv(target_path, index=False)


def run():
  collector = WC2026Collector()

  best_model_res = load_best_config()
  run_wc_2026_monte_carlo(
    iters=iters,
    model_factory=best_model_res.model_factory,
    feats=best_model_res.feats,
    on_complete_iter=collector.on_complete_iter,
    rand_state=rand_state)

  results = [
    (collector.champion, WC_2026_RESULT_CHAMPION_FILE),
    (collector.final, WC_2026_RESULT_FINAL_FILE),
    (collector.semi, WC_2026_RESULT_SEMI_FILE),
    (collector.quarter, WC_2026_RESULT_QUARTER_FILE),
    (collector.last_16, WC_2026_RESULT_16_FILE),
    (collector.last_32, WC_2026_RESULT_32_FILE),
  ]

  run_dir = create_run_dir()
  for counter, result_file in results:
    path = run_dir / result_file
    generate_results(counter, path)

  print(f"Result saved at {run_dir}")


if __name__ == "__main__":
  run()