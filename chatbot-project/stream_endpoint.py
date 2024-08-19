"""Stream the response from the FastAPI query endpoint."""

import requests

url = "http://127.0.0.1:8000/query"

data = {
    "question": (
        "What are some common misconceptions about Henry VIII's wives, and what is the truth behind those myths?"
    )
}


resp = requests.post(url, json=data, stream=True)

# Process and print each line of the streamed response
try:
    for line in resp.iter_lines():
        if line:
            print(line.decode("utf-8"))
finally:
    resp.close()
