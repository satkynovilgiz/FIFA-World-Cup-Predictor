from collections.abc import Callable

import numpy as np

from core.constants import YEARS_COMPLETED
from core.data import DatasetLoader, DataPreparer, RESOURCES_LOADER
from core.features import Feat
from core.world_cup_sim import world_cup_sim_factory, ModelWrapper, WorldCupSimResult


class YearProbCounter:
  def __init__(self, years: list[int] | int):
    if isinstance(years, int):
      years = [years]
    self._counters = {year: {} for year in years}

  def count(self, year: int, teams: list[str] | str):
    if isinstance(teams, str):
      teams = [teams]
    for team in teams:
      if team not in self._counters[year]:
        self._counters[year][team] = 0
      self._counters[year][team] += 1

  def calculate_probs(self, iters: int) -> dict[int, dict[str, float]]:
    probs = {}
    for year, counter in self._counters.items():
      year_probs = {
        team: freq / iters
        for team, freq in counter.items()
      }
      probs[year] = dict(
        sorted(
          year_probs.items(),
          key=lambda item: item[1],
          reverse=True)
      )
    return probs


def run_leave_one_wc_monte_carlo(
        iters: int,
        model_factory: Callable,
        feats: list[Feat]) -> dict[int, dict[str, float]]:
  print(f"Running leave one world cup monte carlo with {iters} iterations...")

  winner_counter = YearProbCounter(YEARS_COMPLETED)

  for year_test in YEARS_COMPLETED:
    print(f"Current year test: {year_test}")
    years_train = [y for y in YEARS_COMPLETED if y != year_test]
    loader = DatasetLoader(years_train, feats=feats, add_reversed_matches=True)
    x, y = loader.load()
    model = model_factory()
    model.fit(x, y)
    model_wrapper = ModelWrapper(model, DataPreparer(RESOURCES_LOADER.load_nations(year_test), feats=feats))
    run_monte_carlo(
      model_wrapper=model_wrapper,
      year=year_test,
      on_complete_iter=lambda res: winner_counter.count(year_test, res.champion),
      iters=iters,
      rng=np.random.default_rng(42))

  return winner_counter.calculate_probs(iters)


def run_wc_2026_monte_carlo(
        iters: int,
        model_factory: Callable,
        feats: list[Feat],
        on_complete_iter: Callable[[WorldCupSimResult], None],
        rand_state: int):
  print(f"Running 2026 world cup monte carlo sim with {iters} iterations...")
  year = 2026
  years_train = YEARS_COMPLETED
  loader = DatasetLoader(years_train, feats=feats, add_reversed_matches=True)
  x, y = loader.load()
  model = model_factory()
  model.fit(x, y)
  model_rapper = ModelWrapper(model, DataPreparer(RESOURCES_LOADER.load_nations(year), feats=feats))
  run_monte_carlo(
    model_wrapper=model_rapper,
    year=year,
    on_complete_iter=on_complete_iter,
    iters=iters,
    rng=np.random.default_rng(rand_state),
  )


def run_monte_carlo(
        model_wrapper: ModelWrapper,
        year: int,
        on_complete_iter: Callable[[WorldCupSimResult], None],
        iters: int,
        rng: np.random.Generator):
  interval_to_log = max(min(iters // 10, 1000), 1)
  for i in range(iters):
    if i % interval_to_log == 0 and i > 0:
      print(f"{i}/{iters} iterations completed")
    sim = world_cup_sim_factory(model_wrapper, year, rng)
    result = sim.run()
    on_complete_iter(result)
