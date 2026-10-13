def test_register_login_me(client):
    response = client.post("/api/v1/auth/register", json={"username": "alice", "email": "alice@example.com", "password": "password123"})
    assert response.status_code == 201
    assert response.get_json()["success"] is True
    me = client.get("/api/v1/auth/me")
    assert me.status_code == 200
    assert me.get_json()["user"]["email"] == "alice@example.com"


def test_reject_short_password(client):
    response = client.post("/api/v1/auth/register", json={"username": "bob", "email": "bob@example.com", "password": "short"})
    assert response.status_code == 400


def test_logout_clears_session(auth_client):
    assert auth_client.get("/api/v1/auth/me").status_code == 200
    response = auth_client.post("/api/v1/auth/logout")
    assert response.status_code == 200
    assert response.get_json()["success"] is True
    me = auth_client.get("/api/v1/auth/me")
    assert me.status_code == 200
    assert me.get_json()["user"]["username"] == "local_workspace"
