import requests

def send_event(payload: dict):

    print("[API] send event:", payload)
    # requests.post("https://your-server/api/events", json=payload)
