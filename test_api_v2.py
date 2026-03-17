import requests
import json

API_BASE_URL = "https://www.tone3000.com/api/v1"
API_AUTH_URL = f"{API_BASE_URL}/auth/session"
API_SEARCH_URL = f"{API_BASE_URL}/tones/search"
TONE3000_API_KEY = "b57ae63a-f178-4d28-b5a2-a7c4a3fb6414"

def test_search(query, gear):
    auth_payload = {"api_key": TONE3000_API_KEY}
    auth_response = requests.post(API_AUTH_URL, json=auth_payload, timeout=10)
    access_token = auth_response.json().get("access_token")
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    search_params = {
        "query": query,
        "gear": gear,
        "page_size": 100
    }
    
    print(f"Testing with query='{query}', gear='{gear}'")
    response = requests.get(API_SEARCH_URL, params=search_params, headers=headers, timeout=15)
    data = response.json()
    items = data.get("data", [])
    print(f"Results Count: {len(items)}")
    if items:
        for i, item in enumerate(items[:5]):
            print(f"Result {i}: {item.get('title')}")
            print(f"  Gear: {item.get('gear')}, Platform: {item.get('platform')}")
            print(f"  Models count: {len(item.get('models', []))}")
            print(f"  Model URL: {item.get('model_url')}")
            print(f"  File Name: {item.get('file_name')}")

if __name__ == "__main__":
    test_search("ampeg", "full-rig")
    print("-" * 20)
    test_search("ampeg", "amp_pedal")
    print("-" * 20)
    test_search("ampeg", "") # No gear filter
