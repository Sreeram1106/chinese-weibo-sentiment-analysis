"""Central configuration: paths, seeds and label metadata."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "labeledData.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.joblib"

SEED = 42

# The dataset ships only numeric label IDs 0-6. The names below are a TENTATIVE
# interpretation inferred from the most discriminative words per class
# (see results/top_features_per_class.csv). They are NOT from official
# documentation and must be described as "tentative" in any report/viva.
LABEL_NAMES = {
    0: "fear",
    1: "disgust",
    2: "happiness",
    3: "surprise",
    4: "like / gratitude",
    5: "sadness",
    6: "anger",
}
LABEL_NAMES_ARE_VERIFIED = False

# Keyword screen used only to build a *rough* COVID-related subset for a
# robustness check. It is not a validated topic classifier.
COVID_TERMS = ["新冠", "疫情", "冠状病毒", "肺炎", "核酸", "抗疫", "疫苗", "接种",
               "口罩", "隔离", "封城", "解封", "病毒", "确诊", "医护", "白衣", "逆行",
               "COVID", "covid"]
