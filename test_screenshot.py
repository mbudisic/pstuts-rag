import requests

url = "https://api.microlink.io"
params = {
    "url": "https://www.framatome.com",
    "screenshot": True,
}

response = requests.get(url, params)

print(response.json())
