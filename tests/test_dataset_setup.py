from zipfile import ZipFile

import pytest

from scripts.prepare_uploaded_datasets import prepare_audio_checkpoint, prepare_datasets, prepare_image_checkpoint


def test_prepare_datasets_extracts_only_expected_csvs(tmp_path):
    cyberbullying_zip = tmp_path / "cyberbullying.zip"
    isot_zip = tmp_path / "isot.zip"
    with ZipFile(cyberbullying_zip, "w") as archive:
        archive.writestr(
            "cyberbullying_tweets.csv",
            "tweet_text,cyberbullying_type\nkind message,not_cyberbullying\n",
        )
        archive.writestr("unrelated.txt", "not extracted")
    with ZipFile(isot_zip, "w") as archive:
        archive.writestr("Fake.csv", "title,text\nmade up,unverified claim\n")
        archive.writestr("True.csv", "title,text\nreported,verified article\n")

    output_dir = tmp_path / "prepared"
    prepared = prepare_datasets(cyberbullying_zip, isot_zip, output_dir=output_dir)

    assert {path.name for path in prepared} == {"cyberbullying_tweets.csv", "Fake.csv", "True.csv"}
    assert not (output_dir / "unrelated.txt").exists()
    with pytest.raises(FileExistsError):
        prepare_datasets(cyberbullying_zip, isot_zip, output_dir=output_dir)


def test_prepare_image_checkpoint_extracts_expected_model(tmp_path):
    archive_path = tmp_path / "models.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("bundle/fake_news_detector/model/deepfake_model.h5", b"keras checkpoint")
        archive.writestr("bundle/audio/backend/deepfake_model.h5", b"audio checkpoint")

    output_path = tmp_path / "weights" / "deepfake_model.h5"
    prepared = prepare_image_checkpoint(archive_path, output_path)

    assert prepared.read_bytes() == b"keras checkpoint"
    with pytest.raises(FileExistsError):
        prepare_image_checkpoint(archive_path, output_path)


def test_prepare_audio_checkpoint_extracts_expected_model(tmp_path):
    archive_path = tmp_path / "models.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("bundle/audio/backend/deepfake_model.h5", b"audio checkpoint")
        archive.writestr("bundle/fake_news_detector/model/deepfake_model.h5", b"image checkpoint")

    output_path = tmp_path / "weights" / "audio_deepfake_model.h5"
    prepared = prepare_audio_checkpoint(archive_path, output_path)

    assert prepared.read_bytes() == b"audio checkpoint"
    with pytest.raises(FileExistsError):
        prepare_audio_checkpoint(archive_path, output_path)