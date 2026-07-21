from typing import Literal, Any

import numpy as np
import pandas as pd

from core.constants import TEAM_1_WIN_ID, TEAM_2_WIN_ID, DRAW_ID, DATA_PATH
from core.features import Feat, prepare_feats_to_match

LABEL_NAME = "class"


def get_class(row: pd.Series) -> int:
  if row["score_1"] > row["score_2"]:
    return TEAM_1_WIN_ID
  if row["score_2"] > row["score_1"]:
    return TEAM_2_WIN_ID
  return DRAW_ID


class ResourcesLoader:
  def __init__(self):
    self._cache: dict[str, Any] = {}

  def load_nations(self, year: int) -> pd.DataFrame:
    path = f"{DATA_PATH}/nations/{year}.csv"
    if path not in self._cache:
      nations = pd.read_csv(path, index_col="id")
      self._cache[path] = nations
    return self._cache[path]


  def load_matches(self, year: int, stage_name: Literal["groups", "kout", "last"]) -> pd.DataFrame:
    path = f"{DATA_PATH}/matches/{year}_{stage_name}.csv"
    if path not in self._cache:
      matches = pd.read_csv(path)
      self._cache[path] = prepare_matches(matches, stage_name)
    return self._cache[path]

  def load_kout_template(self, filename: str) -> pd.DataFrame:
    path = f"{DATA_PATH}/matches/{filename}"
    if path not in self._cache:
      template = pd.read_csv(path)
      self._cache[path] = template
    return self._cache[path]

  def load_third_combinations(self) -> dict[str, int]:
    path = f"{DATA_PATH}/matches/third_combinations.csv"
    if path not in self._cache:
      df = pd.read_csv(path)
      mask_df = df.notna().astype(int)
      third_combinations = {
        ''.join(row.astype(str)): idx
        for idx, row in enumerate(mask_df.values)
      }
      self._cache[path] = third_combinations
    return self._cache[path]

  def load_group_win_vs_thirds(self) -> pd.DataFrame:
    path = f"{DATA_PATH}/matches/group_win_vs_thirds.csv"
    if path not in self._cache:
      df = pd.read_csv(path)
      self._cache[path] = df
    return self._cache[path]


RESOURCES_LOADER = ResourcesLoader()


def prepare_matches(matches: pd.DataFrame, stage_name: str) -> pd.DataFrame:
  if stage_name == "groups":
    matches["stage"] = stage_name
  if "score_1" in matches.columns:
    matches["class"] = matches.apply(get_class, axis=1)
  return matches



class DataPreparer:
  def __init__(self, nations: pd.DataFrame, feats: list[Feat]):
    self._nations = nations
    self._nations_cols = self._nations.columns.to_numpy()
    self._feats = feats

  def prepare_xy(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = matches[LABEL_NAME].to_numpy(dtype=np.int32).ravel()
    return self.prepare_x(matches), y

  def prepare_x(self, matches: pd.DataFrame) -> np.ndarray:
    data = self._merge_team_feats(matches, "team_1")
    data = self._merge_team_feats(data, "team_2")
    data = DataPreparer._calculate_diffs(data)

    prep_feats = prepare_feats_to_match(self._feats)
    return data[prep_feats].to_numpy(dtype=np.float32)

  def _merge_team_feats(self, data: pd.DataFrame, team_id: Literal["team_1", "team_2"]) -> pd.DataFrame:
    col_mapper = {f: f"{team_id}_{f}" for f in self._nations_cols}
    return data.merge(
      self._nations,
      left_on=team_id,
      right_index=True,
      how="left"
    ).rename(columns=col_mapper)

  @staticmethod
  def _calculate_diffs(data: pd.DataFrame) -> pd.DataFrame:
    data[Feat.ATT_DIFF.label] = data["team_1_att"] - data[f"team_2_att"]
    data[Feat.MID_DIFF.label] = data["team_1_mid"] - data[f"team_2_mid"]
    data[Feat.DEF_DIFF.label] = data["team_1_def"] - data[f"team_2_def"]
    data[Feat.ELO_RATING_DIFF.label] = data["team_1_elo_rating"] - data["team_2_elo_rating"]
    data[Feat.TITLES_DIFF.label] = data[f"team_1_titles"] - data[f"team_2_titles"]
    data[Feat.FIFA_RANK_DIFF.label] = data[f"team_2_fifa_rank"] - data[f"team_1_fifa_rank"]

    return data


class DatasetLoader:
  def __init__(self, years: list[int], feats: list[Feat], add_reversed_matches: bool):
    self._years = years
    self._feats = feats
    self._add_reversed_matches = add_reversed_matches

  def load(self) -> tuple[np.ndarray, np.ndarray]:
    x_parts = []
    y_parts = []
    for year in self._years:
      yx, yy = self._load_by_year(year)
      x_parts.append(yx)
      y_parts.extend(yy)

    x = np.vstack(x_parts)
    y = np.asarray(y_parts)

    return x, y

  def _load_by_year(self, year: int) -> tuple[np.ndarray, np.ndarray]:
    nations = RESOURCES_LOADER.load_nations(year)
    group_matches = RESOURCES_LOADER.load_matches(year, "groups")
    group_matches = group_matches.drop(columns="group", axis=1)
    kout_matches = RESOURCES_LOADER.load_matches(year, "kout")
    matches = pd.concat([group_matches, kout_matches], ignore_index=True)

    if self._add_reversed_matches:
      rev = DatasetLoader.get_reversed_matches(matches)
      matches = pd.concat([matches, rev], ignore_index=True)

    return DataPreparer(nations, self._feats).prepare_xy(matches)

  @staticmethod
  def get_reversed_matches(matches: pd.DataFrame) -> pd.DataFrame:
    rev_matches = matches.copy()
    rev_matches["team_1"] = matches["team_2"]
    rev_matches["team_2"] = matches["team_1"]
    rev_matches["score_1"] = matches["score_2"]
    rev_matches["score_2"] = matches["score_1"]
    rev_matches["class"] = rev_matches.apply(get_class, axis=1)
    return rev_matches
