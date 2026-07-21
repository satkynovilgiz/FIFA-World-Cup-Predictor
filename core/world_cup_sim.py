from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from numpy.random import Generator

from core.constants import TEAM_1_WIN_ID, TEAM_2_WIN_ID, DRAW_ID
from core.data import prepare_matches, RESOURCES_LOADER, DataPreparer

GROUPS_STR = "ABCDEFGHIJKL"

class Stage(Enum):
  GROUPS = ("groups", 1)
  ROUND_32 = ("32", 2)
  ROUND_16 = ("16", 3)
  QUARTER = ("quarter", 4)
  SEMI = ("semi", 5)
  THIRD = ("third", 6)
  FINAL = ("final", 7)

  def __new__(cls, label: str, _id: int):
    obj = object.__new__(cls)
    obj.label = label
    obj.id = _id
    return obj

  @staticmethod
  def ids() -> list[int]:
    return [s.id for s in Stage]

  @staticmethod
  def mapped_by_label() -> dict[str, int]:
      return {m.value[0]: m.value[1] for m in Stage}


class Match:
  def __init__(
          self,
          match_num: int,
          team_1: str,
          team_2: str,
          proba: np.ndarray[float],
          rng: Generator):
    self._match_num = match_num
    self._team_1 = team_1
    self._team_2 = team_2
    self._proba = proba
    self._rng = rng
    self._winner = None
    self._loser = None

  @property
  def team_1(self) -> str:
    return self._team_1

  @property
  def team_2(self) -> str:
    return self._team_2

  @property
  def team_1_win_proba(self) -> float:
    return self._proba[TEAM_1_WIN_ID]

  @property
  def team_2_win_proba(self) -> float:
    return self._proba[TEAM_2_WIN_ID]

  @property
  def proba_draw(self) -> float:
    return self._proba[DRAW_ID]

  def play(self):
    clazz = self._rng.choice(
      [DRAW_ID, TEAM_1_WIN_ID, TEAM_2_WIN_ID],
      p=self._proba
    )
    if clazz == TEAM_1_WIN_ID:
      self._winner = self._team_1
      self._loser = self._team_2
    elif clazz == TEAM_2_WIN_ID:
      self._winner = self._team_2
      self._loser = self._team_1

  def tie_breaker(self):
    total = np.sum(self._proba[1:])
    if total == 0:
      no_draw_proba = np.array([0.5, 0.5])
    else:
      no_draw_proba = self._proba[1:] / total

    clazz = self._rng.choice(
      [TEAM_1_WIN_ID, TEAM_2_WIN_ID],
      p=no_draw_proba
    )
    if clazz == TEAM_1_WIN_ID:
      self._winner = self._team_1
      self._loser = self._team_2
    else:
      self._winner = self._team_2
      self._loser = self._team_1

  def get_winner_team(self) -> str:
    return self._winner

  def get_loser_team(self) -> str:
    return self._loser


class ModelWrapper:
  def __init__(self, model, data_preparer: DataPreparer):
    self._model = model
    self._data_prep = data_preparer
    self._groups_cache: dict[int, np.ndarray] = {}
    self._kout_cache: dict[tuple[int, str, str], np.ndarray] = {}

  def predict_groups_proba(self, year: int, matches: pd.DataFrame) -> np.ndarray:
    if year not in self._groups_cache:
      x = self._data_prep.prepare_x(matches)
      pred = self._model.predict_proba(x)
      self._groups_cache[year] = pred
    return self._groups_cache[year]

  def predict_kout_proba(self, year: int, matches: list[dict], stage: Stage) -> np.ndarray:
    pred_result = []
    for match in matches:
      key = year, match["team_1"], match["team_2"]
      if key not in self._kout_cache:
        match_df = prepare_matches(pd.DataFrame([match]), stage.label)
        x = self._data_prep.prepare_x(match_df)
        pred = self._model.predict_proba(x)
        self._kout_cache[key] = pred[0]
      pred_result.append(self._kout_cache[key])
    return np.array(pred_result)


@dataclass
class TeamScoresInGroup:
  points: int = 0
  expected_points: float = 0


class WorldCupGroup:
  def __init__(self, group_id: str):
    self._group_id = group_id
    self._matches: list[Match] = []
    self._teams_scores: dict[str, TeamScoresInGroup] | None = None
    self._teams_rank: list[str] | None = None

  def add_match(self, match: Match):
    self._matches.append(match)

  def complete(self):
    teams = set()
    for m in self._matches:
      m.play()
      teams.add(m.team_1)
      teams.add(m.team_2)

    scores = {t: TeamScoresInGroup() for t in teams}
    for m in self._matches:
      team_1_scores = scores[m.team_1]
      team_2_scores = scores[m.team_2]
      winner = m.get_winner_team()
      if winner:
        scores[winner].points += 3
      else:
        team_1_scores.points += 1
        team_2_scores.points += 1
      team_1_scores.expected_points += 3 * m.team_1_win_proba + m.proba_draw
      team_2_scores.expected_points += 3 * m.team_2_win_proba + m.proba_draw

    scores = dict(
      sorted(
        scores.items(),
        key=lambda item: (item[1].points, item[1].expected_points),
        reverse=True)
    )

    self._teams_scores = scores
    self._teams_rank = [t for t in scores.keys()]

  def get_winner_team(self) -> str:
    return self.get_team_by_position(1)

  def get_runner_up_team(self) -> str:
    return self.get_team_by_position(2)

  def get_third_team(self) -> str:
    return self.get_team_by_position(3)

  def get_scores_by_position(self, pos: int) -> TeamScoresInGroup:
    team = self.get_team_by_position(pos)
    return self._teams_scores[team]

  def get_team_by_position(self, pos: int) -> str:
    return self._teams_rank[pos - 1]


@dataclass(frozen=True)
class WorldCupSimResult:
  champion: str
  runner_up: str
  third: str
  fourth: str
  quarter_finalists: list[str] | None = None
  last_16: list[str] | None = None
  last_32: list[str] | None = None


def _create_complete_result(
        final: dict[int, Match],
        third: dict[int, Match],
        quarter: dict[int, Match],
        round_16: dict[int, Match],
        round_32: dict[int, Match] = None):
  final_match = list(final.values())[0]
  third_match = list(third.values())[0]
  return WorldCupSimResult(
    champion=final_match.get_winner_team(),
    runner_up=final_match.get_loser_team(),
    third=third_match.get_winner_team(),
    fourth=third_match.get_loser_team(),
    quarter_finalists=_get_participants_from_stage(quarter),
    last_16=_get_participants_from_stage(round_16),
    last_32=_get_participants_from_stage(round_32) if round_32 else None
  )


def _create_compact_result(final: dict[int, Match], third: dict[int, Match]):
  final_match = list(final.values())[0]
  third_match = list(third.values())[0]
  return WorldCupSimResult(
    champion=final_match.get_winner_team(),
    runner_up=final_match.get_loser_team(),
    third=third_match.get_winner_team(),
    fourth=third_match.get_loser_team(),
  )


def _get_participants_from_stage(stage: dict[int, Match]) -> list[str]:
  teams = []
  for macth in stage.values():
    teams.append(macth.team_1)
    teams.append(macth.team_2)
  return teams


class BaseWorldCupSim(ABC):
  def __init__(
          self,
          model_wrapper: ModelWrapper,
          year: int,
          kout_template_filename: str,
          rng: Generator):
    self._model_wrapper = model_wrapper
    self._year = year
    self._kout_template = RESOURCES_LOADER.load_kout_template(kout_template_filename)
    self._nations = RESOURCES_LOADER.load_nations(year)
    self._rng = rng

  @abstractmethod
  def run(self) -> WorldCupSimResult:
    pass

  def _play_groups(self) -> dict[str, WorldCupGroup]:
    group_matches = RESOURCES_LOADER.load_matches(self._year, "groups")
    pred = self._model_wrapper.predict_groups_proba(self._year, group_matches)
    groups = {}
    for i, proba in enumerate(pred):
      row = group_matches.iloc[i]
      group_id = row["group"]
      if group_id not in groups:
        groups[group_id] = WorldCupGroup(group_id)
      match = Match(
        match_num=i + 1,
        team_1=row["team_1"],
        team_2=row["team_2"],
        proba=proba,
        rng=self._rng)
      groups[group_id].add_match(match)

    for group in groups.values():
      group.complete()
    return groups


  def _play_quarter(self, round_16_matches: dict[int, Match]) -> dict[int, Match]:
    def get_teams(team_1_origin: str, team_2_origin: str):
      team_1_match_origin = int(team_1_origin.replace("winner match ", ""))
      team_2_match_origin = int(team_2_origin.replace("winner match ", ""))
      match_1 = round_16_matches[team_1_match_origin]
      match_2 = round_16_matches[team_2_match_origin]
      team_1 = match_1.get_winner_team()
      team_2 = match_2.get_winner_team()
      return team_1, team_2

    return self._play_kout_stage(get_teams, Stage.QUARTER)

  def _play_semi(self, quarter_matches: dict[int, Match]) -> dict[int, Match]:
    def get_teams(team_1_origin: str, team_2_origin: str):
      team_1_match_origin = int(team_1_origin.replace("winner match ", ""))
      team_2_match_origin = int(team_2_origin.replace("winner match ", ""))
      match_1 = quarter_matches[team_1_match_origin]
      match_2 = quarter_matches[team_2_match_origin]
      team_1 = match_1.get_winner_team()
      team_2 = match_2.get_winner_team()
      return team_1, team_2

    return self._play_kout_stage(get_teams, Stage.SEMI)

  def _play_third(self, semi_matches: dict[int, Match]) -> dict[int, Match]:
    def get_teams(team_1_origin: str, team_2_origin: str):
      team_1_match_origin = int(team_1_origin.replace("loser match ", ""))
      team_2_match_origin = int(team_2_origin.replace("loser match ", ""))
      match_1 = semi_matches[team_1_match_origin]
      match_2 = semi_matches[team_2_match_origin]
      team_1 = match_1.get_loser_team()
      team_2 = match_2.get_loser_team()
      return team_1, team_2

    return self._play_kout_stage(get_teams, Stage.THIRD)

  def _play_final(self, semi_matches: dict[int, Match]) -> dict[int, Match]:
    def get_teams(team_1_origin: str, team_2_origin: str):
      team_1_match_origin = int(team_1_origin.replace("winner match ", ""))
      team_2_match_origin = int(team_2_origin.replace("winner match ", ""))
      match_1 = semi_matches[team_1_match_origin]
      match_2 = semi_matches[team_2_match_origin]
      team_1 = match_1.get_winner_team()
      team_2 = match_2.get_winner_team()
      return team_1, team_2

    return self._play_kout_stage(get_teams, Stage.FINAL)

  def _play_kout_stage(self, get_teams: Callable[[str, str], tuple[str, str]], stage: Stage) -> dict[int, Match]:
    template = self._kout_template[self._kout_template["stage"] == stage.label]
    matches = []
    for _, row in template.iterrows():
      team_1_origin = row["team_1"]
      team_2_origin = row["team_2"]
      team_1, team_2 = get_teams(team_1_origin, team_2_origin)
      matches.append({
        "match_num": row["match_num"],
        "stage": stage.label,
        "team_1": team_1,
        "team_2": team_2
      })

    pred: np.ndarray = self._model_wrapper.predict_kout_proba(self._year, matches, stage)

    result: dict[int, Match] = {}
    for i, proba in enumerate(pred):
      row = matches[i]
      match_num = row["match_num"]
      match = Match(
        match_num=row["match_num"],
        team_1=row["team_1"],
        team_2=row["team_2"],
        proba=proba,
        rng=self._rng)
      match.play()
      if not match.get_winner_team():
        match.tie_breaker()
      result[match_num] = match

    return result



class WorldCupSim32(BaseWorldCupSim):
  def __init__(self, model_wrapper: ModelWrapper, year: int, rng: Generator):
    super().__init__(
      model_wrapper=model_wrapper,
      year=year,
      kout_template_filename="kout_template_32.csv",
      rng=rng)

  def run(self) -> WorldCupSimResult:
    groups = self._play_groups()
    round_16 = self._play_round_16(groups)
    quarter = self._play_quarter(round_16)
    semi = self._play_semi(quarter)
    third = self._play_third(semi)
    final = self._play_final(semi)

    return _create_compact_result(final, third)


  def _play_round_16(self, groups: dict[str, WorldCupGroup]) -> dict[int, Match]:
    def get_teams(team_1_origin: str, team_2_origin: str):
      team_1_group_id = team_1_origin.replace("winner group ", "")
      team_2_group_id = team_2_origin.replace("runner-up group ", "")
      team_1_group = groups[team_1_group_id]
      team_2_group = groups[team_2_group_id]
      team_1 = team_1_group.get_winner_team()
      team_2 = team_2_group.get_runner_up_team()
      return team_1, team_2

    return self._play_kout_stage(get_teams, Stage.ROUND_16)



class WorldCupSim48(BaseWorldCupSim):
  def __init__(self, model_wrapper: ModelWrapper, year: int, rng: Generator):
    super().__init__(
      model_wrapper=model_wrapper,
      year=year,
      kout_template_filename="kout_template_48.csv",
      rng=rng)
    self._third_combinations = RESOURCES_LOADER.load_third_combinations()
    self._group_win_vs_thirds = RESOURCES_LOADER.load_group_win_vs_thirds()

  def run(self):
    groups = self._play_groups()
    round_32 = self._play_round_32(groups)
    round_16 = self._play_round_16(round_32)
    quarter = self._play_quarter(round_16)
    semi = self._play_semi(quarter)
    third = self._play_third(semi)
    final = self._play_final(semi)

    return _create_complete_result(
      final=final,
      third=third,
      quarter=quarter,
      round_16=round_16,
      round_32=round_32)

  def _play_round_32(self, groups: dict[str, WorldCupGroup]) -> dict[int, Match]:
    win_vs_third_comb = self._get_win_vs_third_combination(groups)
    def get_teams(team_1_origin: str, team_2_origin: str):
      if team_1_origin.startswith("winner group"):
        team_1_group_id = team_1_origin.replace(r"winner group ", "")
        team_1 = groups[team_1_group_id].get_winner_team()
      else: # runner-up
        team_1_group_id = team_1_origin.replace(r"runner-up group ", "")
        team_1 = groups[team_1_group_id].get_runner_up_team()

      if team_2_origin.startswith("3rd"):
        team_2_group_id = win_vs_third_comb[team_1_group_id]
        team_2 = groups[team_2_group_id].get_third_team()
      else: # group runner-up
        team_2_group_id = team_2_origin.replace("runner-up group ", "")
        team_2 = groups[team_2_group_id].get_runner_up_team()

      return team_1, team_2

    return self._play_kout_stage(get_teams, Stage.ROUND_32)

  def _get_win_vs_third_combination(self, groups: dict[str, WorldCupGroup]):
    thirds = []
    for group_id, group in groups.items():
      scores = group.get_scores_by_position(3)
      thirds.append((group_id, scores))

    thirds.sort(key=lambda item: (item[1].points, item[1].expected_points), reverse=True)
    best_thirds = thirds[0:8]

    best_thirds.sort(key=lambda item: item[0])
    best_thirds_groups = set([item[0] for item in best_thirds])

    comb = ''.join('1' if c in best_thirds_groups else '0' for c in GROUPS_STR)

    comb_row_idx = self._third_combinations[comb]
    return self._group_win_vs_thirds.iloc[comb_row_idx]


  def _play_round_16(self, round_32_matches: dict[int, Match]) -> dict[int, Match]:
    def get_teams(team_1_origin: str, team_2_origin: str):
      team_1_match_origin = int(team_1_origin.replace("winner match ", ""))
      team_2_match_origin = int(team_2_origin.replace("winner match ", ""))
      match_1 = round_32_matches[team_1_match_origin]
      match_2 = round_32_matches[team_2_match_origin]
      team_1 = match_1.get_winner_team()
      team_2 = match_2.get_winner_team()
      return team_1, team_2

    return self._play_kout_stage(get_teams, Stage.ROUND_16)


def world_cup_sim_factory(model_wrapper: ModelWrapper, year: int, rng: Generator) -> BaseWorldCupSim:
  if year == 2026:
    return WorldCupSim48(model_wrapper, year, rng)
  return WorldCupSim32(model_wrapper, year, rng)
