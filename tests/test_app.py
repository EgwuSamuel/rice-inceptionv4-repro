import io
import sqlite3

import numpy as np
import pytest
from PIL import Image

from app import create_app


class FakeClassifier:
    def __init__(self, conf):
        self.conf = conf

    def predict(self, img):
        rest = (1 - self.conf) / 3
        probs = {"Bacterial leaf blight": self.conf, "Brown spot": rest, "Healthy": rest, "Leaf smut": rest}
        return {"label": "Bacterial leaf blight", "confidence": self.conf, "probabilities": probs}


def _client(tmp_path, conf):
    app = create_app(
        FakeClassifier(conf),
        {"TESTING": True, "DATABASE": str(tmp_path / "t.sqlite"),
         "UPLOAD_DIR": str(tmp_path / "up"), "SECRET_KEY": "test"},
    )
    return app.test_client()


def _png():
    buf = io.BytesIO()
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(buf, "PNG")
    buf.seek(0)
    return buf


def _login(c):
    assert c.post("/register", data={"username": "farmer", "password": "longpassword"}).status_code == 302
    assert c.post("/login", data={"username": "farmer", "password": "wrongpassword"}).status_code == 401
    assert c.post("/login", data={"username": "farmer", "password": "longpassword"}).status_code == 302


def test_requires_login(tmp_path):
    assert _client(tmp_path, 0.9).get("/predict").status_code == 302


def test_password_is_hashed(tmp_path):
    c = _client(tmp_path, 0.9)
    _login(c)
    row = sqlite3.connect(tmp_path / "t.sqlite").execute("SELECT password_hash FROM users").fetchone()
    assert row[0] != "longpassword" and row[0].startswith(("scrypt:", "pbkdf2:"))


def test_prediction_with_advice_and_history(tmp_path):
    c = _client(tmp_path, 0.95)
    _login(c)
    r = c.post("/predict", data={"file": (_png(), "../../evil name.png")},
               content_type="multipart/form-data")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "Bacterial leaf blight" in html and "resistant varieties" in html
    saved = list((tmp_path / "up").iterdir())
    assert len(saved) == 1 and saved[0].suffix == ".png" and "evil" not in saved[0].name
    assert "Bacterial leaf blight" in c.get("/history").get_data(as_text=True)


def test_low_confidence_abstains(tmp_path):
    c = _client(tmp_path, 0.40)
    _login(c)
    html = c.post("/predict", data={"file": (_png(), "leaf.png")},
                  content_type="multipart/form-data").get_data(as_text=True)
    assert "Uncertain result" in html and "resistant varieties" not in html


@pytest.mark.parametrize("payload,name", [(b"not an image", "x.png"), (b"abc", "x.exe")])
def test_rejects_bad_uploads(tmp_path, payload, name):
    c = _client(tmp_path, 0.9)
    _login(c)
    r = c.post("/predict", data={"file": (io.BytesIO(payload), name)}, content_type="multipart/form-data")
    assert r.status_code == 400
