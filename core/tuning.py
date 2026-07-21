import ast
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from itertools import product, combinations

import pandas as pd

import core.models as models
from core.constants import GRID_SEARCH_RESULT_PATH
from core.evaluation import leave_one_world_cup_score, tournament_score
from core.features import FEATS_GROUPS, Feat, has_main_feature, feats_from_str_list
from core.models import MODEL_NAMES_MAP

rf_grid = {
  "n_estimators": [100, 200, 300],
  "max_depth": [4, 6, 8, None],
  "min_samples_leaf": [1, 2, 4, 8],
  "max_features": ["sqrt", 0.5]
}

xgb_grid = {
  "n_estimators": [50, 100, 200],
  "max_depth": [2, 3, 4],
  "learning_rate": [0.01, 0.03, 0.1],
  "subsample": [0.7, 0.9, 1.0],
  "colsample_bytree": [0.7, 1.0],
  "min_child_weight": [1, 3, 5]
}

lr_grid = {
  "C": [ 0.001, 0.01, 0.1, 1, 10, 100]
}

lgbm_grid = {
  "n_estimators": [50, 100, 200],
  "learning_rate": [0.01, 0.03, 0.1],
  "num_leaves": [7, 15, 31],
  "max_depth": [2, 3, 4],
  "min_child_samples": [5, 10, 20]
}

cat_grid = {
  "iterations": [50, 75, 100, 200],
  "depth": [1, 2, 3, 4, 5],
  "learning_rate": [0.03, 0.05, 0.1],
  "l2_leaf_reg": [4, 7, 10, 12]
}


GRID_MAP = {
  "xgb": (models.build_xgb, xgb_grid),
  "lr": (models.build_lr, lr_grid),
  "rf": (models.build_rf, rf_grid),
  "lgbm": (models.build_lgbm, lgbm_grid),
  "cat": (models.build_cat, cat_grid),
}


def get_feat_combinations() -> list[list[Feat]]:
  comb_feats = []
  for num in range(1, len(FEATS_GROUPS) + 1):
    for c in combinations(FEATS_GROUPS, num):
      feats = []
      for gf in c:
        feats.extend(gf)
      if has_main_feature(feats):
        comb_feats.append(feats)

  return comb_feats


class GridSearch:
  def __init__(self, model_id: str, model_builder: Callable, param_grid: dict[list]):
    self._model_id = model_id
    self._model_builder = model_builder
    self._param_grid = param_grid
    self._feat_combinations = get_feat_combinations()

  def run(self) -> pd.DataFrame:
    print(f"Running {self._model_id} grid search...")
    params_values = self._get_param_values_combinations()
    print(f"Num of param combinations: {len(params_values)}")

    keys = list(self._param_grid.keys())
    results = []
    for i, values in enumerate(params_values):
      init = datetime.now()
      params = dict(zip(keys, values))
      print(f"params: {params}")

      results.extend(self._run_for_feats(params))

      print(f"{i + 1}/{len(params_values)}")
      print(f"Completed in {(datetime.now() - init).total_seconds()}")

    print("Done!")
    return pd.DataFrame(results).sort_values("avg_loss", ascending=True)


  def _get_param_values_combinations(self):
    params_values = []
    for values in product(*self._param_grid.values()):
      params_values.append(values)
    return params_values

  def _run_for_feats(self, params):
    results = []
    for feats in self._feat_combinations:
      feats_str = [ft.label for ft in feats]
      print(f"feats: {feats_str}")
      score = leave_one_world_cup_score(
        lambda: self._model_builder(params),
        feats
      )

      results.append({
        "params": json.dumps(params),
        "feats": feats_str,
        **score
      })

    return results


@dataclass(frozen=True)
class Combination:
  params: dict
  feats: list[str]
  avg_loss: float


class TournamentEvaluator:
  def __init__(self, model_id: str, model_builder: Callable):
    self._model_id = model_id
    self._model_builder = model_builder

  def run(self) -> list[dict]:
    print(f"Running {self._model_id} tournament evaluation...")
    grid_search_res = pd.read_csv(
      f"{GRID_SEARCH_RESULT_PATH}/{self._model_id}.csv").sort_values("avg_loss")
    combs = self._get_top_combinations(grid_search_res)
    mc_iters = 10_000
    result = []
    for i, comb in enumerate(combs):
      init = datetime.now()

      feats_str = comb.feats
      feats = feats_from_str_list(feats_str)
      model_factory = lambda: self._model_builder(comb.params)
      score = tournament_score(iters=mc_iters, model_factory=model_factory, feats=feats)

      result.append({
        "model_id": self._model_id,
        "model_name": MODEL_NAMES_MAP[self._model_id],
        "mc_iters": mc_iters,
        **score,
        "params": json.dumps(comb.params),
        "feats": comb.feats,
        "avg_loss": comb.avg_loss,
      })

      print(f"{self._model_id}: {i + 1}/{len(combs)} completed!")
      print(f"Took {(datetime.now() - init).total_seconds()} secs")

    return result


  @staticmethod
  def _get_top_combinations(grid_search_res: pd.DataFrame, top_n: int = 3) -> list[Combination]:
    combs = []
    for feats, group in grid_search_res.groupby("feats"):
      feats_list = ast.literal_eval(feats)

      if not has_main_feature(feats_from_str_list(feats_list)):
        continue

      best_row = group.nsmallest(1, "avg_loss").iloc[0]

      params = json.loads(best_row["params"])

      combs.append(Combination(
        params=params,
        feats=feats_list,
        avg_loss=best_row["avg_loss"]
      ))

    combs.sort(key=lambda c: c.avg_loss)

    return combs[:top_n]
