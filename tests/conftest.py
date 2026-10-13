import pytest
import sys
import os
import shutil
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "instance" / "test-temp"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(TEST_TEMP_ROOT)
os.environ["TEMP"] = str(TEST_TEMP_ROOT)
os.environ["TMPDIR"] = str(TEST_TEMP_ROOT)
tempfile.tempdir = str(TEST_TEMP_ROOT)

from app import create_app
from app.config import TestingConfig
from app.extensions import db


@pytest.fixture()
def tmp_path(request):
    safe_name = "".join(character if character.isalnum() else "_" for character in request.node.name)
    path = TEST_TEMP_ROOT / f"{safe_name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def app():
    app = create_app(TestingConfig)
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def auth_client(client):
    client.post("/api/v1/auth/register", json={"username": "tester", "email": "tester@example.com", "password": "password123"})
    return client
