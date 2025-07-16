from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
import google.generativeai as genai

video_id = "xDQL3vWwcp0" # replace with any youtube video id
transcript = YouTubeTranscriptApi.get_transcript(video_id)
transcript_joined = " ".join([line['text'] for line in transcript])

api_key = "GEMINI_API_KEY" # Replace with your actual API key
prompt = f'Summarize this youtube video transcript and also give result with punctuations \ntext = "{transcript_joined}."'

genai.configure(api_key=api_key)
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=prompt
)

response = model.generate_content("now give summary of this video transcript")
reply = response.text
print(reply)

