import argparse
import csv
import shutil
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "datasets" / "raw" / "user_uploads"
IMAGE_MODEL_MEMBER = "fake_news_detector/model/deepfake_model.h5"
DEFAULT_IMAGE_MODEL_PATH = (
    ROOT_DIR
    / "model_weights"
    / "external_sources"
    / "ShieldVision-Multimodal-Detection"
    / "fake_news_detector"
    / "model"
    / "deepfake_model.h5"
)
DEFAULT_AUDIO_MODEL_PATH = (
    ROOT_DIR
    / "model_weights"
    / "external_sources"
    / "ShieldVision-Multimodal-Detection"
    / "audio"
    / "backend"
    / "deepfake_model.h5"
)


def _find_member(archive, filename):
    matches = [
        name for name in archive.namelist()
        if PurePosixPath(name).name == filename and not name.endswith("/")
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {filename} file in archive; found {len(matches)}.")
    return matches[0]


def _validate_csv_member(archive, member, required_columns):
    with archive.open(member) as binary:
        rows = csv.reader(line.decode("utf-8-sig") for line in binary)
        header = next(rows, [])
    missing = set(required_columns) - set(header)
    if missing:
        raise ValueError(f"{PurePosixPath(member).name} is missing columns: {', '.join(sorted(missing))}")


def prepare_datasets(
    cyberbullying_zip,
    isot_zip,
    output_dir=DEFAULT_OUTPUT_DIR,
    sentiment_train=None,
    sentiment_test=None,
    overwrite=False,
):
    if bool(sentiment_train) != bool(sentiment_test):
        raise ValueError("Provide both sentiment Parquet splits or neither.")

    output_dir = Path(output_dir)
    archive_specs = [
        (Path(cyberbullying_zip), "cyberbullying_tweets.csv", ("tweet_text", "cyberbullying_type")),
        (Path(isot_zip), "Fake.csv", ("title", "text")),
        (Path(isot_zip), "True.csv", ("title", "text")),
    ]
    validated_archives = []
    for archive_path, filename, required_columns in archive_specs:
        try:
            with ZipFile(archive_path) as archive:
                member = _find_member(archive, filename)
                _validate_csv_member(archive, member, required_columns)
        except (OSError, BadZipFile) as error:
            raise ValueError(f"Could not read {archive_path}: {error}") from error
        validated_archives.append((archive_path, member, filename))

    files = [(archive_path, member, output_dir / filename) for archive_path, member, filename in validated_archives]
    if sentiment_train and sentiment_test:
        for source, filename in (
            (Path(sentiment_train), "train-00000-of-00001.parquet"),
            (Path(sentiment_test), "test-00000-of-00001.parquet"),
        ):
            try:
                frame = pd.read_parquet(source, columns=["text", "annotation"])
            except (OSError, ValueError, ImportError) as error:
                raise ValueError(f"Invalid sentiment Parquet file {source}: {error}") from error
            if frame.empty or not {"positive", "negative"}.issubset(set(frame["annotation"].dropna().astype(str))):
                raise ValueError(f"{source} must contain both positive and negative annotations.")
            files.append((source, None, output_dir / filename))

    existing = [destination for _, _, destination in files if destination.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing dataset files: {paths}. Pass --overwrite to replace them.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for source, member, destination in files:
        if member is None:
            shutil.copyfile(source, destination)
        else:
            with ZipFile(source) as archive, archive.open(member) as input_file, destination.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file)
        print(f"Prepared {destination}")
    return [destination for _, _, destination in files]


def prepare_image_checkpoint(archive_path, output_path=DEFAULT_IMAGE_MODEL_PATH, overwrite=False):
    archive_path = Path(archive_path)
    output_path = Path(output_path)
    try:
        with ZipFile(archive_path) as archive:
            matches = [
                name for name in archive.namelist()
                if name.endswith(IMAGE_MODEL_MEMBER) and not name.endswith("/")
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one {IMAGE_MODEL_MEMBER} checkpoint; found {len(matches)}.")
            member = matches[0]
            if output_path.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite {output_path}. Pass --overwrite to replace it.")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as input_file, output_path.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file)
    except FileExistsError:
        raise
    except (OSError, BadZipFile) as error:
        raise ValueError(f"Could not extract image checkpoint from {archive_path}: {error}") from error
    print(f"Prepared local image checkpoint {output_path}; verify source license and accuracy before deployment.")
    return output_path


def prepare_audio_checkpoint(archive_path, output_path=DEFAULT_AUDIO_MODEL_PATH, overwrite=False):
    archive_path = Path(archive_path)
    output_path = Path(output_path)
    member_suffix = "audio/backend/deepfake_model.h5"
    try:
        with ZipFile(archive_path) as archive:
            matches = [
                name for name in archive.namelist()
                if name.endswith(member_suffix) and not name.endswith("/")
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one {member_suffix} checkpoint; found {len(matches)}.")
            member = matches[0]
            if output_path.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite {output_path}. Pass --overwrite to replace it.")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as input_file, output_path.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file)
    except FileExistsError:
        raise
    except (OSError, BadZipFile) as error:
        raise ValueError(f"Could not extract audio checkpoint from {archive_path}: {error}") from error
    print(f"Prepared local audio checkpoint {output_path}; verify source license and accuracy before deployment.")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare local training files without committing raw datasets.")
    parser.add_argument("--cyberbullying-zip", required=True, help="Archive containing cyberbullying_tweets.csv")
    parser.add_argument("--isot-zip", required=True, help="Archive containing Fake.csv and True.csv")
    parser.add_argument("--sentiment-train", help="Optional labeled sentiment training Parquet file")
    parser.add_argument("--sentiment-test", help="Optional held-out sentiment test Parquet file")
    parser.add_argument("--image-model-archive", help="Optional archive containing the ShieldVision Keras image model")
    parser.add_argument("--audio-model-archive", help="Optional archive containing the ShieldVision Keras audio model")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Replace files already in the output directory")
    args = parser.parse_args()
    try:
        prepare_datasets(
            args.cyberbullying_zip,
            args.isot_zip,
            output_dir=args.output_dir,
            sentiment_train=args.sentiment_train,
            sentiment_test=args.sentiment_test,
            overwrite=args.overwrite,
        )
        if args.image_model_archive:
            prepare_image_checkpoint(args.image_model_archive, overwrite=args.overwrite)
        if args.audio_model_archive:
            prepare_audio_checkpoint(args.audio_model_archive, overwrite=args.overwrite)
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()