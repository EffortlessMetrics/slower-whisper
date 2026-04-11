from fastapi import FastAPI
from fastapi.testclient import TestClient
from transcription.service_middleware import add_security_headers

app = FastAPI()
app.middleware("http")(add_security_headers)

@app.get("/")
def read_root():
    return {"message": "Hello World"}

client = TestClient(app)

def test_security_headers():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"

    csp = response.headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "cdn.jsdelivr.net" in csp
    assert "fastapi.tiangolo.com" in csp
