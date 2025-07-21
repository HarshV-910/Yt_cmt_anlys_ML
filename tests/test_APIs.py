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
#     payload = {"video_id": "VJgdOMXhEj0"} 
    
#     response = requests.post(GEMINI_API_URL, json=payload)
    
#     # Assert the status code
#     assert response.status_code == 200
    
#     # Assert that the response contains the 'summary' key
#     response_data = response.json()
#     assert "summary" in response_data
#     assert isinstance(response_data["summary"], str) # Ensure the summary is a string
#     assert len(response_data["summary"]) > 0 # Ensure the summary is not empty

API_BASE = "http://localhost:5000"  # Make sure your Flask app is running here

@pytest.mark.integration
def test_summarize_video_endpoint():
    video_id = "VJgdOMXhEj0"
    payload = {"video_id": video_id}

    response = requests.post(f"{API_BASE}/summarize_video", json=payload)

    # Check for 200 OK status
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"

    response_data = response.json()

    # Validate structure
    assert "summary" in response_data, "'summary' key not found in response"
    assert isinstance(response_data["summary"], str), "Summary should be a string"
    assert len(response_data["summary"].strip()) > 0, "Summary should not be empty"
