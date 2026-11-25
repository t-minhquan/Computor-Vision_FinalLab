import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

import pytest

from src.training.package_colab import package_colab


def test_colab_bundle_contains_code_data_and_pinned_versions(tmp_path, image_bytes):
    data = tmp_path / "raw/train/happy"
    data.mkdir(parents=True)
    (data / "sample.png").write_bytes(image_bytes)
    output = tmp_path / "bundle.zip"
    report = package_colab(data.parent, output)
    assert report["training_images"] == 1
    with ZipFile(output) as bundle:
        assert bundle.testzip() is None
        names = bundle.namelist()
        assert "src/training/train.py" in names
        assert "data/raw/train/happy/sample.png" in names
        assert all(not name.startswith(("models/", "data/raw/test/", ".venv/")) for name in names)
        manifest = json.loads(bundle.read("colab_manifest.json"))
        requirements = bundle.read("requirements-colab.txt").decode()
        assert manifest["training_images"] == 1
        for name, value in manifest["dependencies"].items():
            assert f"{name}=={value}" in requirements
    with pytest.raises(FileExistsError):
        package_colab(data.parent, output)


def test_colab_bundle_rejects_empty_data(tmp_path):
    with pytest.raises(ValueError, match="No labeled"):
        package_colab(tmp_path, tmp_path / "empty.zip")


def test_colab_notebook_code_compiles_and_invokes_training_script():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/01_train_fer_colab.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"][0]["cell_type"] == "markdown"
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, "colab_cell", "exec")
            assert isinstance(cell["outputs"], list)
            code.append(source)
    combined = "\n".join(code)
    assert '"src.training.train"' in combined
    assert '"--cropped-faces"' in combined
    assert "drive.mount" in combined and "files.download" in combined
    assert ".fit(" not in combined
    for index in (8, 10):
        source = "".join(notebook["cells"][index]["source"])
        assert "str(TRAIN_PYTHON)" in source
        assert "sys.executable" not in source
    train_source = "".join(notebook["cells"][8]["source"])
    assert 'Path(TRAIN_ENV) / "bin/python"' in train_source
    assert "/opt/py312" not in train_source


def _notebook_cell(index):
    path = Path(__file__).resolve().parents[1] / "notebooks/01_train_fer_colab.ipynb"
    return "".join(json.loads(path.read_text(encoding="utf-8"))["cells"][index]["source"])


def _mock_upload(monkeypatch, upload):
    colab = ModuleType("google.colab")
    colab.files = SimpleNamespace(upload=upload)
    monkeypatch.setitem(sys.modules, "google", ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)


def _make_colab_bundle(tmp_path, image_bytes):
    data = tmp_path / "input/happy"
    data.mkdir(parents=True)
    (data / "sample.png").write_bytes(image_bytes)
    output = tmp_path / "fer2013_colab.zip"
    package_colab(data.parent, output)
    return output


def test_notebook_reuses_existing_zip_and_extracted_project(tmp_path, image_bytes, monkeypatch):
    _make_colab_bundle(tmp_path, image_bytes)

    def unexpected_upload():
        raise AssertionError("Existing bundle should not need another upload.")

    _mock_upload(monkeypatch, unexpected_upload)
    source = _notebook_cell(2).replace(
        'CONTENT_DIR = Path("/content")', f"CONTENT_DIR = Path({str(tmp_path)!r})"
    )
    namespace = {}
    exec(source, namespace)
    assert namespace["manifest"]["training_images"] == 1
    marker = tmp_path / "fer_lab/preserve.txt"
    marker.write_text("keep this file")
    exec(source, namespace)
    assert marker.read_text() == "keep this file"


def test_notebook_prompts_for_upload_when_zip_is_missing(tmp_path, image_bytes, monkeypatch):
    archive = _make_colab_bundle(tmp_path, image_bytes)
    archive_bytes = archive.read_bytes()
    archive.unlink()
    calls = []

    def upload():
        calls.append(True)
        selected = tmp_path / "selected_bundle.zip"
        selected.write_bytes(archive_bytes)
        return {selected.name: archive_bytes}

    _mock_upload(monkeypatch, upload)
    source = _notebook_cell(2).replace(
        'CONTENT_DIR = Path("/content")', f"CONTENT_DIR = Path({str(tmp_path)!r})"
    )
    exec(source, {})
    assert calls == [True]
    assert (tmp_path / "fer_lab/src/training/train.py").is_file()


def test_notebook_rejects_archive_path_traversal(tmp_path, image_bytes, monkeypatch):
    archive = _make_colab_bundle(tmp_path, image_bytes)
    with ZipFile(archive, "a") as bundle:
        bundle.writestr("../escaped.txt", "unsafe")
    _mock_upload(monkeypatch, lambda: {})
    source = _notebook_cell(2).replace(
        'CONTENT_DIR = Path("/content")', f"CONTENT_DIR = Path({str(tmp_path)!r})"
    )
    with pytest.raises(ValueError):
        exec(source, {})
    assert not (tmp_path / "escaped.txt").exists()
    assert not (tmp_path / "fer_lab").exists()


def test_python313_kernel_creates_python312_training_environment(tmp_path, monkeypatch):
    import subprocess

    project = tmp_path / "fer_lab"
    project.mkdir()
    calls = []

    def run(command, **kwargs):
        assert kwargs["check"] is True
        calls.append(command)
        if "venv" in command:
            assert command[command.index("--python") + 1] == "3.12"
            interpreter = Path(command[-1]) / "bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()

    def check_output(command, **kwargs):
        assert "fer_train_py312" in command[0]
        if "sys.version_info.major" in command[-1]:
            return "3.12\n"
        return json.dumps({"python_version": "3.12.6", "dependencies": {}})

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "check_output", check_output)
    namespace = {
        "sys": SimpleNamespace(executable="kernel-python3.13"),
        "CONTENT_DIR": tmp_path,
        "PROJECT_DIR": project,
        "manifest": {"python_major_minor": ["3", "12"]},
    }
    exec(_notebook_cell(4), namespace)
    assert calls[0][:4] == ["kernel-python3.13", "-m", "pip", "install"]
    assert calls[0][-1] == "uv"
    assert calls[-1][0] == str(namespace["TRAIN_PYTHON"])
    assert (
        json.loads((project / "training_environment.json").read_text())["python_version"]
        == "3.12.6"
    )
    exec(_notebook_cell(4), namespace)
    assert sum("venv" in command for command in calls) == 1


def test_training_cell_reports_missing_setup_without_overwriting_log(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("previous run")
    with pytest.raises(RuntimeError, match="Run step 2"):
        exec(_notebook_cell(8), {"RUN_DIR": tmp_path})
    with pytest.raises(RuntimeError, match="Rerun step 2"):
        exec(_notebook_cell(8), {"TRAIN_ENV": tmp_path, "Path": Path, "RUN_DIR": tmp_path})
    assert log.read_text() == "previous run"


def test_training_cell_uses_setup_interpreter_instead_of_stale_variable(tmp_path, monkeypatch):
    env = tmp_path / "environment"
    interpreter = env / "bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    commands = []

    def popen(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=iter(["Extracting neutral: 2 images\n"]), wait=lambda: 0)

    progress = SimpleNamespace(
        update=lambda *args: None,
        close=lambda: None,
        refresh=lambda: None,
    )

    def bar(**kwargs):
        return progress

    bar.write = lambda *args: None
    monkeypatch.setitem(sys.modules, "tqdm", ModuleType("tqdm"))
    monkeypatch.setitem(sys.modules, "tqdm.auto", SimpleNamespace(tqdm=bar))
    namespace = {
        "Path": Path,
        "TRAIN_ENV": env,
        "TRAIN_PYTHON": "/opt/py312/bin/python",
        "PROJECT_DIR": tmp_path,
        "RUN_DIR": tmp_path,
        "MODEL_PATH": tmp_path / "fer.joblib",
        "subprocess": SimpleNamespace(Popen=popen, PIPE=-1, STDOUT=-2),
    }
    exec(_notebook_cell(8), namespace)
    assert commands[0][0] == str(interpreter)
    assert namespace["return_code"] == 0
    assert "Extracting neutral" in (tmp_path / "train.log").read_text()
