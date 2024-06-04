import requests
from base64 import b64encode
import soundfile as sf
import io
# The endpoint URL
url = 'http://localhost:5001/stt'

# base64 encode image so it can be passed in json
with open("../novice/audio/audio_25.wav", "rb") as f:
    audio_bytes = f.read()
    audio_input, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    print(len(audio_input))
    audio = b64encode(audio_bytes).decode("utf-8")

# Construct the payload as expected by the FastAPI endpoint
data = {
    "instances": [
        {
            "b64": audio,
        }
    ]
}

# Sending a POST request
response = requests.post(url, json=data)

# Print the response from the server
print("Status Code:", response.status_code)
print("Response:", response.json())
