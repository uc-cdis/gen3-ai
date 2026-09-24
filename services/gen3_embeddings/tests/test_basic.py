def test_version(client):
    """GET /_version/ returns 200 with a non-empty version string."""
    response = client.get("/_version/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert isinstance(data["version"], str)


def test_status(client):
    """GET /_status/ returns 200 with status OK and a numeric timestamp."""
    response = client.get("/_status/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert isinstance(data["timestamp"], float)
