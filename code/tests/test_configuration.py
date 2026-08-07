from pathlib import Path

import pytest
from neurodxfm.configuration import apply_override, deep_merge, load_settings, parse_override


def test_deep_merge() -> None:
    base = {"model": {"dimension": 32, "depth": 2}, "seed": 1}
    override = {"model": {"depth": 4}}
    assert deep_merge(base, override) == {"model": {"dimension": 32, "depth": 4}, "seed": 1}


def test_parse_numeric_override() -> None:
    path, value = parse_override("optimizer.learning_rate=0.001")
    assert path == ["optimizer", "learning_rate"]
    assert value == 0.001


def test_parse_string_override() -> None:
    path, value = parse_override("runtime.device=cuda")
    assert path == ["runtime", "device"]
    assert value == "cuda"


def test_parse_override_requires_equals() -> None:
    with pytest.raises(ValueError):
        parse_override("runtime.device")


def test_apply_nested_override() -> None:
    configuration: dict[str, object] = {"runtime": {"seed": 1}}
    apply_override(configuration, "runtime.seed=2")
    assert configuration["runtime"] == {"seed": 2}


def test_apply_override_creates_path() -> None:
    configuration: dict[str, object] = {}
    apply_override(configuration, "runtime.seed=2")
    assert configuration == {"runtime": {"seed": 2}}


def test_apply_override_rejects_scalar_traversal() -> None:
    configuration: dict[str, object] = {"runtime": 1}
    with pytest.raises(ValueError):
        apply_override(configuration, "runtime.seed=2")


def test_load_settings_defaults_from_sections(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(
        "model: {}\nloss: {}\noptimizer: {}\nschedule: {}\nruntime: {}\ndata: {}\n",
        encoding="utf-8",
    )
    settings = load_settings(path)
    assert settings.model.representation_dim == 768
    assert settings.schedule.epochs == 200


def test_load_settings_override(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(
        "model: {}\nloss: {}\noptimizer: {}\nschedule: {}\nruntime: {}\ndata: {}\n",
        encoding="utf-8",
    )
    settings = load_settings(path, ["runtime.seed=99"])
    assert settings.runtime.seed == 99
