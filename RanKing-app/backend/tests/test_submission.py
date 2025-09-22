import io
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_upload_valid_image():
    file_data = io.BytesIO(b"fakeimagebytes")
    response = client.post(
        "/submission/upload?user_id=test123",
        files={"file": ("test.jpg", file_data, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["status"] in ["accepted", "rejected"]

def test_upload_invalid_format():
    file_data = io.BytesIO(b"fakeimagebytes")
    response = client.post(
        "/submission/upload?user_id=test123",
        files={"file": ("test.gif", file_data, "image/gif")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid file format"

def test_upload_oversized_file():
    big_data = io.BytesIO(b"0" * (11 * 1024 * 1024))  # 11MB
    response = client.post(
        "/submission/upload?user_id=test123",
        files={"file": ("big.jpg", big_data, "image/jpeg")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "File too large"
