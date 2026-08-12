import requests

BASE_URL = "http://127.0.0.1:5000"

def test_flow():
    # 1. Test Product Creation
    product_payload = {
        "product_name": "Test Laptop Box",
        "weight_kg": 3.2
    }
    res = requests.post(f"{BASE_URL}/api/product", json=product_payload)
    print("Product Creation Response:", res.status_code, res.json())
    
    if res.status_code == 200:
        product_id = res.json().get("product_id")
        
        # 2. Test Recommendation API
        reco_payload = {"product_id": product_id}
        reco_res = requests.post(f"{BASE_URL}/api/recommend", json=reco_payload)
        print("Recommendation Response:", reco_res.status_code, reco_res.json())

if __name__ == "__main__":
    test_flow()