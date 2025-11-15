import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_prompt_empty():
    response = client.get("/getPrompt")
    assert response.status_code == 200
    assert "prompt" in response.json()

def test_save_and_get_prompt():
    # Save a new prompt
    test_prompt = "This is a test prompt"
    response = client.post("/chngPrompt", json={"prompt": test_prompt})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["prompt"] == test_prompt

    # Get the saved prompt
    response = client.get("/getPrompt")
    assert response.status_code == 200
    assert response.json()["prompt"] == test_prompt
