from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

TEMPLATES_DIR = BASE_DIR / "templates"
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
GENERATED_DIR = STATIC_DIR / "generated"

TRAIN_CODE_DIR = PROJECT_DIR / "train_code"
MODELS_DIR = PROJECT_DIR / "models"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)



SERVICE_TITLE = "Сервис многоэтапного анализа инсульта на КТ головного мозга"

DISCLAIMER_TEXT = (
    "Результат работы системы носит вспомогательный характер, "
    "не заменяет заключение врача-рентгенолога и должен интерпретироваться "
    "с учетом клинических данных. Модель может ошибаться."
)



# train code files

BINARY_TRAIN_CODE_PATH = TRAIN_CODE_DIR / "train_8_2_new.py"
TYPE_TRAIN_CODE_PATH = TRAIN_CODE_DIR / "train_type_classifier.py"
BLEEDING_TRAIN_CODE_PATH = TRAIN_CODE_DIR / "train_bleeding_segmenter.py"
ISCHEMIA_STAGE1_TRAIN_CODE_PATH = TRAIN_CODE_DIR / "train_ischemia_coarse_localizer.py"
ISCHEMIA_STAGE2_TRAIN_CODE_PATH = TRAIN_CODE_DIR / "train_ischemia_refinement.py"


# model directories

BINARY_MODEL_DIR = MODELS_DIR / "binary_classifier"
TYPE_MODEL_DIR = MODELS_DIR / "type_classifier"
BLEEDING_MODEL_DIR = MODELS_DIR / "bleeding_segmenter"
ISCHEMIA_STAGE1_MODEL_DIR = MODELS_DIR / "ischemia_stage1"
ISCHEMIA_STAGE2_MODEL_DIR = MODELS_DIR / "ischemia_stage2"


# model files

BINARY_MODEL_PATH = BINARY_MODEL_DIR / "best_model.pth"
BINARY_THRESHOLDS_PATH = BINARY_MODEL_DIR / "thresholds.json"

TYPE_MODEL_PATH = TYPE_MODEL_DIR / "best_model.pth"
TYPE_CLASS_MAPPING_PATH = TYPE_MODEL_DIR / "class_mapping.json"

BLEEDING_MODEL_PATH = BLEEDING_MODEL_DIR / "best_model.pth"
BLEEDING_THRESHOLDS_PATH = BLEEDING_MODEL_DIR / "thresholds.json"

ISCHEMIA_STAGE1_MODEL_PATH = ISCHEMIA_STAGE1_MODEL_DIR / "best_model.pth"
ISCHEMIA_STAGE1_THRESHOLDS_PATH = ISCHEMIA_STAGE1_MODEL_DIR / "thresholds.json"

ISCHEMIA_STAGE2_MODEL_PATH = ISCHEMIA_STAGE2_MODEL_DIR / "best_model.pth"
ISCHEMIA_STAGE2_THRESHOLDS_PATH = ISCHEMIA_STAGE2_MODEL_DIR / "thresholds.json"




CLASS_LABELS_RU = {
    0: "Признаки инсульта не выявлены",
    1: "Геморрагический инсульт",
    2: "Ишемический инсульт",
}

CLASS_LABELS_EN = {
    0: "Normal",
    1: "Bleeding",
    2: "Ischemia",
}




ROUTE_LABELS_RU = {
    "stopped_by_binary": "первичным скринингом",
    "type_normal": "классификацией типа инсульта",
    "bleeding_segmenter_positive": "сегментацией геморрагического очага",
    "bleeding_segmenter_negative": "сегментацией геморрагического очага",
    "ischemia_stage1_stage2_positive": "двухэтапной сегментацией ишемического очага",
    "ischemia_stage1_stage2_negative": "двухэтапной сегментацией ишемического очага",
}



# inference settings

USE_ISCHEMIA_TTA = True



REQUIRED_FILES = {
    "binary_model": BINARY_MODEL_PATH,
    "binary_thresholds": BINARY_THRESHOLDS_PATH,
    "type_model": TYPE_MODEL_PATH,
    "bleeding_model": BLEEDING_MODEL_PATH,
    "bleeding_thresholds": BLEEDING_THRESHOLDS_PATH,
    "ischemia_stage1_model": ISCHEMIA_STAGE1_MODEL_PATH,
    "ischemia_stage1_thresholds": ISCHEMIA_STAGE1_THRESHOLDS_PATH,
    "ischemia_stage2_model": ISCHEMIA_STAGE2_MODEL_PATH,
    "ischemia_stage2_thresholds": ISCHEMIA_STAGE2_THRESHOLDS_PATH,
    "binary_train_code": BINARY_TRAIN_CODE_PATH,
    "type_train_code": TYPE_TRAIN_CODE_PATH,
    "bleeding_train_code": BLEEDING_TRAIN_CODE_PATH,
    "ischemia_stage1_train_code": ISCHEMIA_STAGE1_TRAIN_CODE_PATH,
    "ischemia_stage2_train_code": ISCHEMIA_STAGE2_TRAIN_CODE_PATH,
}
