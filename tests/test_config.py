"""Configuration lookup in checkouts and installed environments."""

from energy_forecast.config.settings import project_root


def test_project_root_uses_working_directory_with_a_config(monkeypatch, tmp_path):
    """An installed package inside Docker must read /app/configs, not site-packages."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("project: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert project_root() == tmp_path
