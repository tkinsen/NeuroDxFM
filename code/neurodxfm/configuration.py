import dataclasses
import json
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from .settings import ExperimentSettings

T = TypeVar("T")


def _convert(value: object, annotation: object) -> object:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if annotation is Path:
        return Path(str(value))
    if origin is tuple:
        inner = arguments[0] if arguments else Any
        return tuple(_convert(item, inner) for item in value)
    if origin in (Union, UnionType):
        candidates = [candidate for candidate in arguments if candidate is not type(None)]
        if value is None:
            return None
        return _convert(value, candidates[0]) if candidates else value
    if dataclasses.is_dataclass(annotation):
        return instantiate(annotation, value)
    return value


def instantiate(cls: type[T], values: object) -> T:
    if not isinstance(values, dict):
        raise TypeError("configuration section must be a mapping")
    hints = get_type_hints(cls)
    fields = {field.name: field for field in dataclasses.fields(cls)}
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(f"unknown configuration values: {sorted(unknown)}")
    converted = {name: _convert(value, hints[name]) for name, value in values.items()}
    return cls(**converted)


def deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def parse_override(value: str) -> tuple[list[str], object]:
    if "=" not in value:
        raise ValueError("override must contain equals")
    key, raw = value.split("=", 1)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return key.split("."), parsed


def apply_override(configuration: dict[str, object], override: str) -> None:
    path, value = parse_override(override)
    cursor = configuration
    for key in path[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"override path traverses scalar: {key}")
        cursor = child
    cursor[path[-1]] = value


def load_settings(path: Path, overrides: list[str] | None = None) -> ExperimentSettings:
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise TypeError("configuration root must be a mapping")
    for override in overrides or []:
        apply_override(values, override)
    return instantiate(ExperimentSettings, values)


def save_settings(settings: ExperimentSettings, path: Path) -> None:
    values = dataclasses.asdict(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(values, handle, sort_keys=False)
