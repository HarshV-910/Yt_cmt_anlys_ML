# # tests/test_APIs.py
# import pytest
# import requests

# # Replace with actual endpoints if deployed externally
# FLASK_API_URL = "http://127.0.0.1:5000/analyze"
# GEMINI_API_URL = "http://127.0.0.1:5000/gemini"


# def test_flask_api():
#     payload = {"comment": "This is a great video!"}
#     response = requests.post(FLASK_API_URL, json=payload)
#     assert response.status_code == 200
#     assert "sentiment" in response.json()


# def test_gemini_api():
#     payload = {"query": "Summarize this comment thread"}
#     response = requests.post(GEMINI_API_URL, json=payload)
#     assert response.status_code == 200
#     assert "response" in response.json()
