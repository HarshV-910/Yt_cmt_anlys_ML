# tests/test_APIs.py
import pytest
import requests

# Replace with actual endpoints if deployed externally
FLASK_API_URL = "http://localhost:5000/predict"
# GEMINI_API_URL = "http://localhost:5000/summarize_video"

def test_flask_api():
    payload = {
        "comments": ["this is worst video","this is very good thing","this is neutral"]
    }
    response = requests.post(FLASK_API_URL, json=payload)
    
    # Assert the status code first
    assert response.status_code == 200
    
    # Get the JSON response
    response_data = response.json()
    
    # Assert that the response is a list
    assert isinstance(response_data, list)
    
    # Assert that the list is not empty
    assert len(response_data) > 0
    
    # Assert that each item in the list is a dictionary and contains 'comment' and 'sentiment' keys
    for item in response_data:
        assert isinstance(item, dict)
        assert "comment" in item
        assert "sentiment" in item



# def test_gemini_api():
#     payload = {"video_id": "VJgdOMXhEj0"}  # Valid video with transcript

#     response = requests.post(GEMINI_API_URL, json=payload)

#     assert response.status_code == 200

#     response_data = response.json()
#     assert "summary" in response_data
#     assert isinstance(response_data["summary"], str)
#     assert len(response_data["summary"]) > 0

