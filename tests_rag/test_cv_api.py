"""
tests_rag/test_cv_api.py
Kiểm thử các FastAPI Endpoints của phân hệ Sàng lọc CV (CV Screening API).
Bao gồm:
- GET /api/v1/cv/sample-criteria
- POST /api/v1/cv/screen-text
- POST /api/v1/cv/screen (Single & Batch File Upload)
- Bắt lỗi ngoại lệ và validation (400, 415, 422)
"""

import io
import json
import pytest
from starlette.testclient import TestClient
from server import app

SAMPLE_CV_1 = """
NGUYỄN VĂN AN
Email: an.nguyen@example.com | SĐT: 0912345678
Địa chỉ: Hà Nội
KỸ NĂNG: Python, FastAPI, PostgreSQL, Docker, LangChain
KINH NGHIỆM:
Công ty AI (2021 - 2024) - Senior Backend Engineer
- Xây dựng hệ thống RAG nội bộ tra cứu tài liệu với FastAPI và PostgreSQL.
HỌC VẤN: Đại học Bách Khoa Hà Nội (2015 - 2020)
"""

SAMPLE_CV_2 = """
LÊ THỊ BÌNH
Email: binh.le@example.com | SĐT: 0987654321
Địa chỉ: TP.HCM
KỸ NĂNG: JavaScript, React, HTML, CSS, TailwindCSS
KINH NGHIỆM:
Công ty Web XYZ (2022 - 2024) - Frontend Developer
- Phát triển giao diện người dùng bằng React.
HỌC VẤN: Đại học Khoa học Tự nhiên TP.HCM (2018 - 2022)
"""


@pytest.fixture
def client():
    return TestClient(app)


def test_get_sample_criteria(client):
    response = client.get("/api/v1/cv/sample-criteria")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 3
    assert any("Senior Backend" in c["job_title"] for c in data)


def test_screen_cv_text_endpoint(client):
    payload = {
        "cv_text": SAMPLE_CV_1,
        "criteria": {
            "job_title": "Senior Backend Developer",
            "min_total_experience_years": 2.0,
            "required_skills": ["Python", "FastAPI"],
            "preferred_skills": ["Docker", "LangChain"],
            "passing_score": 70.0,
        },
        "file_name": "an_nguyen_cv.txt",
    }
    response = client.post("/api/v1/cv/screen-text", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["job_title"] == "Senior Backend Developer"
    assert data["candidate_info"]["full_name"] == "NGUYỄN VĂN AN"
    assert data["overall_score"] >= 80.0
    assert data["is_passed"] is True
    assert len(data["key_strengths"]) > 0
    assert len(data["interview_questions"]) >= 3
    assert data["metadata"]["file_name"] == "an_nguyen_cv.txt"


def test_screen_cv_file_single(client):
    criteria_json = json.dumps({
        "job_title": "Python & Backend Developer",
        "required_skills": ["Python", "FastAPI"],
        "passing_score": 70.0,
    })

    file_bytes = SAMPLE_CV_1.encode("utf-8")
    files = [("files", ("nguyen_van_an.txt", io.BytesIO(file_bytes), "text/plain"))]
    data = {"criteria": criteria_json}

    response = client.post("/api/v1/cv/screen", data=data, files=files)
    assert response.status_code == 200
    res_data = response.json()

    assert res_data["total_cvs"] == 1
    assert res_data["pass_count"] == 1
    assert res_data["top_candidate"] == "NGUYỄN VĂN AN"
    assert len(res_data["results"]) == 1
    assert res_data["results"][0]["overall_score"] >= 80.0


def test_screen_cv_batch_files_ranking(client):
    """Kiểm tra sàng lọc nhiều hồ sơ cùng lúc và xếp hạng điểm số chính xác."""
    criteria_json = json.dumps({
        "job_title": "Senior Python Engineer",
        "required_skills": ["Python", "FastAPI", "PostgreSQL"],
        "preferred_skills": ["LangChain"],
        "passing_score": 70.0,
    })

    files = [
        ("files", ("ung_vien_frontend.txt", io.BytesIO(SAMPLE_CV_2.encode("utf-8")), "text/plain")),
        ("files", ("ung_vien_python.txt", io.BytesIO(SAMPLE_CV_1.encode("utf-8")), "text/plain")),
    ]
    data = {"criteria": criteria_json}

    response = client.post("/api/v1/cv/screen", data=data, files=files)
    assert response.status_code == 200
    res_data = response.json()

    assert res_data["total_cvs"] == 2
    # Ứng viên 1 (Python) phải có điểm cao hơn và đứng đầu danh sách xếp hạng
    assert res_data["top_candidate"] == "NGUYỄN VĂN AN"
    assert res_data["results"][0]["overall_score"] > res_data["results"][1]["overall_score"]
    assert res_data["pass_count"] == 1
    assert res_data["fail_count"] == 1


def test_screen_cv_unsupported_file_extension(client):
    criteria_json = json.dumps({"job_title": "DevOps", "required_skills": ["Linux"]})
    files = [("files", ("malicious.exe", io.BytesIO(b"binary data"), "application/octet-stream"))]
    data = {"criteria": criteria_json}

    response = client.post("/api/v1/cv/screen", data=data, files=files)
    assert response.status_code == 415


def test_screen_cv_invalid_criteria_json(client):
    files = [("files", ("cv.txt", io.BytesIO(SAMPLE_CV_1.encode("utf-8")), "text/plain"))]
    data = {"criteria": "{invalid_json_format"}

    response = client.post("/api/v1/cv/screen", data=data, files=files)
    assert response.status_code == 422
