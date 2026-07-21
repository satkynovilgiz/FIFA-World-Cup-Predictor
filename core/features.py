from enum import Enum


class Feat(Enum):
  ATT_DIFF = ("att_diff", "Difference in attacking rating between both teams (EA Sports Fifa game rating).", 1)
  MID_DIFF = ("mid_diff", "Difference in midfield rating between both teams (EA Sports Fifa game rating).", 2)
  DEF_DIFF = ("def_diff", "Difference in defensive rating between both teams (EA Sports Fifa game rating).", 3)
  ELO_RATING_DIFF = ("elo_rating_diff", "Difference in Elo ratings between both teams.", 4)
  FIFA_RANK_DIFF = ("fifa_rank_diff", "Difference in FIFA ranking positions.", 5)
  TITLES_DIFF = ("titles_diff", "Difference in number of World Cup titles won.", 6)

  def __new__(cls, label: str, description: str, order: int):
    obj = object.__new__(cls)
    obj.label = label
    obj.description = description
    obj.order = order
    return obj

MAIN_FEATURES = {
    Feat.ATT_DIFF,
    Feat.MID_DIFF,
    Feat.DEF_DIFF,
    Feat.ELO_RATING_DIFF,
}

FEATS_GROUPS = [
  [Feat.ATT_DIFF, Feat.MID_DIFF, Feat.DEF_DIFF],
  [Feat.ELO_RATING_DIFF],
  [Feat.TITLES_DIFF],
  [Feat.FIFA_RANK_DIFF],
]

FEATS_BY_VALUE = {ft.label: ft for ft in Feat}


def feats_from_str_list(feats_str: list[str]) -> list[Feat]:
  return [FEATS_BY_VALUE[ft] for ft in feats_str]


def has_main_feature(feats: list[Feat]):
  return any(ft in MAIN_FEATURES for ft in feats)


def prepare_feats_to_match(feats: list[Feat]) -> list[str]:
  sorted_feats: list[Feat] = sorted(feats, key=lambda f: f.order)
  return [f.label for f in sorted_feats]
