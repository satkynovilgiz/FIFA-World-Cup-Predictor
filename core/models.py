import ast
import json
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from core.constants import TOURNAMENT_EVAL_PATH
from core.features import Feat, feats_from_str_list


def build_rf(params):
  base_model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    min_samples_leaf=params["min_samples_leaf"],
    max_features=params["max_features"],
    random_state=42,
    n_jobs=1
  )
  return CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=3
  )


def build_xgb(params):
  return XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    learning_rate=params["learning_rate"],
    subsample=params["subsample"],
    colsample_bytree=params["colsample_bytree"],
    min_child_weight=params["min_child_weight"],
    random_state=42
  )


def build_lr(params):
  return LogisticRegression(
    C=params["C"],
    max_iter=5000
  )


def build_lgbm(params):
  return LGBMClassifier(
    objective="multiclass",
    n_estimators=params["n_estimators"],
    learning_rate=params["learning_rate"],
    num_leaves=params["num_leaves"],
    max_depth=params["max_depth"],
    min_child_samples=params["min_child_samples"],
    random_state=42,
    verbose=-1,
    n_jobs=1,
  )


def build_cat(params):
  return CatBoostClassifier(
    loss_function="MultiClass",
    iterations=params["iterations"],
    depth=params["depth"],
    learning_rate=params["learning_rate"],
    l2_leaf_reg=params["l2_leaf_reg"],
    verbose=False,
    random_seed=42
  )

MODEL_BUILDERS_MAP = {
  "xgb": build_xgb,
  "lr": build_lr,
  "rf": build_rf,
  "lgbm": build_lgbm,
  "cat": build_cat,
}

MODEL_NAMES_MAP = {
  "xgb": "XGBoost",
  "lr": "Logistic Regression",
  "rf": "Random Forest",
  "lgbm": "LightGBM",
  "cat": "CatBoost",
}


@dataclass(frozen=True)
class BestModelConfig:
  model_id: str
  model_name: str
  model_factory: Callable
  feats: list[Feat]


def load_best_config() -> BestModelConfig:
  df = pd.read_csv(TOURNAMENT_EVAL_PATH)
  best_row = df.iloc[0]
  model_id = best_row["model_id"]
  model_builder = MODEL_BUILDERS_MAP[model_id]
  params = json.loads(best_row["params"])
  model_factory = lambda: model_builder(params)
  feats = feats_from_str_list(ast.literal_eval(best_row["feats"]))

  return BestModelConfig(
    model_id=model_id,
    model_name=best_row["model_name"],
    model_factory=model_factory,
    feats=feats)
