import requests
from base64 import b64encode

# The endpoint URL
url = 'http://localhost:5004/identify'

# base64 encode image so it can be passed in json
with open("../novice/images/image_1.jpg", "rb") as f:
    image = b64encode(f.read()).decode("utf-8")

# Construct the payload as expected by the FastAPI endpoint
data = {
    "instances": [
        {
            "b64": image,
            "caption": "blue commercial aircraft"
        }
    ]
}

# Sending a POST request
response = requests.post(url, json=data)

# Print the response from the server
print("Status Code:", response.status_code)
print("Response:", response.json())
