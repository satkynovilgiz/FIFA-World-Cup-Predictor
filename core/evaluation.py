from collections.abc import Callable

import numpy as np
from sklearn.metrics import log_loss, accuracy_score

from core.constants import YEARS_COMPLETED, REAL_CHAMPIONS, DRAW_ID, TEAM_1_WIN_ID, TEAM_2_WIN_ID
from core.data import DatasetLoader
from core.features import Feat
from core.monte_carlo import run_leave_one_wc_monte_carlo


EPSILON = 1e-15


def get_sample_weights(num_matches: int):
  wc_32_num_matches = 64
  if num_matches != wc_32_num_matches:
    raise Exception(f"Invalid num macthes {num_matches}")
  return [1 if i < 48 else 2 for i in range(num_matches)]


def leave_one_world_cup_score(model_factory, feats: list[Feat]):
  losses = []
  accs = []

  for year_test in YEARS_COMPLETED:
    years_train = [
      y for y in YEARS_COMPLETED
      if y != year_test
    ]

    x_train, y_train = DatasetLoader(years_train, feats, add_reversed_matches=True).load()

    model = model_factory()
    model.fit(x_train, y_train)

    x_test, y_test = DatasetLoader([year_test], feats, add_reversed_matches=False).load()

    probs = model.predict_proba(x_test)
    preds = model.predict(x_test)

    loss = log_loss(
      y_test,
      probs,
      labels=[DRAW_ID, TEAM_1_WIN_ID, TEAM_2_WIN_ID],
      sample_weight=get_sample_weights(len(y_test)),
    )
    losses.append(loss)

    accs.append(accuracy_score(y_test,preds))

  return {
    "avg_loss": np.mean(losses),
    "avg_acc": np.mean(accs)
  }



def tournament_score(iters: int, model_factory: Callable, feats: list[Feat]) -> dict:
  probs = run_leave_one_wc_monte_carlo(iters=iters, model_factory=model_factory, feats=feats)

  first_four_probs = {}
  pred_champions = {}
  for y, y_probs in probs.items():
    first_four_teams = list(y_probs.keys())[0:4]
    first_four_probs[y] = {
      team: y_probs[team]
      for team in first_four_teams
    }
    pred_champions[y] = first_four_teams[0]

  champion_in_first_four_count = 0
  hit_champion_count = 0
  real_champions_probs_in_pred = []
  real_champions_probs = {}
  for y, team in REAL_CHAMPIONS.items():
    if team in first_four_probs[y]:
      champion_in_first_four_count += 1
    if team == pred_champions[y]:
      hit_champion_count += 1

    real_champion_prob = probs[y].get(team, 0)
    real_champions_probs_in_pred.append(
      max(real_champion_prob, EPSILON)
    )
    real_champions_probs[y] = real_champion_prob

  tournament_loss = -np.mean(np.log(real_champions_probs_in_pred))

  return {
    "tournament_loss": tournament_loss,
    "champion_in_first_four_count": champion_in_first_four_count,
    "hit_champion_count": hit_champion_count,
    **{f"{y}_pod": pod for y, pod in first_four_probs.items()},
    "real_champion_probs": real_champions_probs,
  }
