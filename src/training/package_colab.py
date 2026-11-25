"""Package the current training code, dependency versions and FER2013 train images."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_PACKAGES = (
    "numpy",
    "scikit-learn",
    "scipy",
    "joblib",
    "Pillow",
    "opencv-python-headless",
)


def package_colab(data_dir: Path, output: Path) -> dict:
    if not data_dir.is_dir():
        raise ValueError(f"Training directory does not exist: {data_dir}")
    images = sorted(
        p
        for p in data_dir.glob("*/*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError("No labeled training images found.")
    manifest = {
        "python_major_minor": list(platform.python_version_tuple()[:2]),
        "python_version": platform.python_version(),
        "dependencies": {name: version(name) for name in TRAINING_PACKAGES},
        "training_images": len(images),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves an earlier bundle if this command is run twice.
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as bundle:
        for source in sorted((PROJECT_ROOT / "src").rglob("*.py")):
            bundle.write(source, source.relative_to(PROJECT_ROOT).as_posix())
        for image in images:
            bundle.write(image, "data/raw/train/" + image.relative_to(data_dir).as_posix())
        bundle.writestr("colab_manifest.json", json.dumps(manifest, indent=2))
        bundle.writestr(
            "requirements-colab.txt",
            "\n".join(f"{name}=={value}" for name, value in manifest["dependencies"].items())
            + "\n",
        )
    return {"bundle": str(output.resolve()), "training_images": len(images)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/raw/train")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / ".tmp/fer2013_colab.zip")
    args = parser.parse_args()
    try:
        print(json.dumps(package_colab(args.data_dir, args.output), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
