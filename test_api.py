import requests
import json

API_BASE_URL = "https://www.tone3000.com/api/v1"
API_AUTH_URL = f"{API_BASE_URL}/auth/session"
API_SEARCH_URL = f"{API_BASE_URL}/tones/search"
TONE3000_API_KEY = "b57ae63a-f178-4d28-b5a2-a7c4a3fb6414"

def test_search():
    auth_payload = {"api_key": TONE3000_API_KEY}
    auth_response = requests.post(API_AUTH_URL, json=auth_payload, timeout=10)
    access_token = auth_response.json().get("access_token")
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    search_params = {
        "query": "ampeg",
        "gear": "full-rig"
    }
    
    response = requests.get(API_SEARCH_URL, params=search_params, headers=headers, timeout=15)
    data = response.json().get("data", [])
    for i, item in enumerate(data[:3]):
        print(f"Result {i} keys: {list(item.keys())}")
        print(f"  Platform: {item.get('platform')}")
        print(f"  Gear: {item.get('gear')}")
        if "models" in item:
            print(f"  Models: {len(item['models'])}")
        if "model_url" in item:
            print(f"  Model URL: {item['model_url']}")
        if "file_name" in item:
            print(f"  File Name: {item['file_name']}")

if __name__ == "__main__":
    test_search()
