import importlib
import json
import sys
from dataclasses import dataclass

import torch

from app.config import (
    BINARY_MODEL_PATH,
    BINARY_THRESHOLDS_PATH,
    BLEEDING_MODEL_PATH,
    BLEEDING_THRESHOLDS_PATH,
    BLEEDING_TRAIN_CODE_PATH,
    BINARY_TRAIN_CODE_PATH,
    ISCHEMIA_STAGE1_MODEL_PATH,
    ISCHEMIA_STAGE1_THRESHOLDS_PATH,
    ISCHEMIA_STAGE1_TRAIN_CODE_PATH,
    ISCHEMIA_STAGE2_MODEL_PATH,
    ISCHEMIA_STAGE2_THRESHOLDS_PATH,
    ISCHEMIA_STAGE2_TRAIN_CODE_PATH,
    REQUIRED_FILES,
    TRAIN_CODE_DIR,
    TYPE_CLASS_MAPPING_PATH,
    TYPE_MODEL_PATH,
    TYPE_TRAIN_CODE_PATH,
)


@dataclass
class ModelRegistry:
    device: torch.device
    binary_module: object
    type_module: object
    bleeding_module: object
    stage1_module: object
    stage2_module: object
    binary_model: object
    type_model: object
    bleeding_model: object
    stage1_model: object
    stage2_model: object
    binary_thresholds: dict
    bleeding_thresholds: dict
    stage1_thresholds: dict
    stage2_thresholds: dict
    type_class_mapping: dict


_REGISTRY: ModelRegistry | None = None


def _ensure_required_files():
    missing = [name for name, path in REQUIRED_FILES.items() if not path.exists()]
    if missing:
        missing_lines = "\n".join(f"- {name}: {REQUIRED_FILES[name]}" for name in missing)
        raise FileNotFoundError(
            "Не найдены обязательные файлы для запуска сервиса:\n"
            f"{missing_lines}"
        )


def _ensure_train_code_importable():
    train_code_str = str(TRAIN_CODE_DIR)
    if train_code_str not in sys.path:
        sys.path.insert(0, train_code_str)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_class_mapping(raw_mapping):
    return {int(k): str(v) for k, v in raw_mapping.items()}


def get_model_registry() -> ModelRegistry:
    global _REGISTRY

    if _REGISTRY is not None:
        return _REGISTRY

    _ensure_required_files()
    _ensure_train_code_importable()

    binary_module = importlib.import_module("train_8_2_new")
    type_module = importlib.import_module("train_type_classifier")
    bleeding_module = importlib.import_module("train_bleeding_segmenter")
    stage1_module = importlib.import_module("train_ischemia_coarse_localizer")
    stage2_module = importlib.import_module("train_ischemia_refinement")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    binary_model = binary_module.build_model(pretrained=False)
    binary_model.load_state_dict(torch.load(BINARY_MODEL_PATH, map_location=device))
    binary_model = binary_model.to(device)
    binary_model.eval()

    type_model = type_module.build_model(pretrained=False)
    type_model.load_state_dict(torch.load(TYPE_MODEL_PATH, map_location=device))
    type_model = type_model.to(device)
    type_model.eval()

    bleeding_model = bleeding_module.build_model(pretrained=False)
    bleeding_model.load_state_dict(torch.load(BLEEDING_MODEL_PATH, map_location=device))
    bleeding_model = bleeding_model.to(device)
    bleeding_model.eval()

    stage1_model = stage1_module.build_model(pretrained=False)
    stage1_model.load_state_dict(torch.load(ISCHEMIA_STAGE1_MODEL_PATH, map_location=device))
    stage1_model = stage1_model.to(device)
    stage1_model.eval()

    stage2_model = stage2_module.build_model(pretrained=False)
    stage2_model.load_state_dict(torch.load(ISCHEMIA_STAGE2_MODEL_PATH, map_location=device))
    stage2_model = stage2_model.to(device)
    stage2_model.eval()

    _REGISTRY = ModelRegistry(
        device=device,
        binary_module=binary_module,
        type_module=type_module,
        bleeding_module=bleeding_module,
        stage1_module=stage1_module,
        stage2_module=stage2_module,
        binary_model=binary_model,
        type_model=type_model,
        bleeding_model=bleeding_model,
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        binary_thresholds=_load_json(BINARY_THRESHOLDS_PATH),
        bleeding_thresholds=_load_json(BLEEDING_THRESHOLDS_PATH),
        stage1_thresholds=_load_json(ISCHEMIA_STAGE1_THRESHOLDS_PATH),
        stage2_thresholds=_load_json(ISCHEMIA_STAGE2_THRESHOLDS_PATH),
        type_class_mapping=_normalize_class_mapping(_load_json(TYPE_CLASS_MAPPING_PATH)),
    )

    return _REGISTRY
