from app.core.config import Settings, get_settings


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check(client):
    settings = get_settings()
    response = client.get(f"{settings.API_V1_PREFIX}/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_name"] == settings.APP_NAME
    assert body["version"] == settings.APP_VERSION


def test_cors_origins_parsed_from_comma_string(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://a.com, http://b.com")
    settings = Settings(_env_file=None)
    assert settings.CORS_ORIGINS == ["http://a.com", "http://b.com"]


def test_cors_header_for_allowed_origin(client):
    origin = get_settings().CORS_ORIGINS[0]
    response = client.get("/", headers={"Origin": origin})
    assert response.headers.get("access-control-allow-origin") == origin
