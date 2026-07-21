from pathlib import Path

DRAW_ID = 0
TEAM_1_WIN_ID = 1
TEAM_2_WIN_ID = 2
YEARS_COMPLETED = [2006, 2010, 2014, 2018, 2022]

PROJECT_ROOT = Path(__file__).parent.parent

DATA_PATH = f"{PROJECT_ROOT}/data"
OUTPUT_PATH = f"{PROJECT_ROOT}/output"
GRID_SEARCH_RESULT_PATH = f"{OUTPUT_PATH}/grid_search"
TOURNAMENT_EVAL_PATH = f"{OUTPUT_PATH}/tournament_evaluation.csv"
SELECTED_MODEL_RESULT = f"{OUTPUT_PATH}/selected_model_result.csv"

WC_2026_RESULT_DIR = f"{OUTPUT_PATH}/world_cup_2026_results"
WC_2026_RESULT_32_FILE = "last_32.csv"
WC_2026_RESULT_16_FILE = "last_16.csv"
WC_2026_RESULT_QUARTER_FILE = "quarter.csv"
WC_2026_RESULT_SEMI_FILE = "semi.csv"
WC_2026_RESULT_FINAL_FILE = "final.csv"
WC_2026_RESULT_CHAMPION_FILE = "champion.csv"

REAL_CHAMPIONS = {
  2006: 'italy',
  2010: 'spain',
  2014: 'germany',
  2018: 'france',
  2022: 'argentina',
}
