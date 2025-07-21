# tests/test_APIs.py
import pytest
import requests

# Replace with actual endpoints if deployed externally
FLASK_API_URL = "http://localhost:5000/predict"
# GEMINI_API_URL = "http://localhost:5000/summarize_video" # Updated to your new endpoint

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
        # Optionally, you can also check the type of sentiment if you have specific expectations
        # assert isinstance(item["sentiment"], int) # or str, depending on your output


# def test_gemini_api():
#     # You'll need a valid YouTube video ID for a successful test.
#     # This ID should correspond to a video with an available transcript.
#     # For testing purposes, you might use a known short video ID with a transcript.
#     # Example: "dQw4w9WgXcQ" (Rick Astley - Never Gonna Give You Up) often works for transcript tests.
#     payload = {"video_id": "5_50z2nL1g0"} 
    
#     response = requests.post(GEMINI_API_URL, json=payload)
    
#     # Assert the status code
#     assert response.status_code == 200
    
#     # Assert that the response contains the 'summary' key
#     response_data = response.json()
#     assert "summary" in response_data
#     assert isinstance(response_data["summary"], str) # Ensure the summary is a string
#     assert len(response_data["summary"]) > 0 # Ensure the summary is not empty