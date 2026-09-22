import json
import urllib.request
import time

API_URL = "http://127.0.0.1:8001/api/v1/search"

queries = [
    {"label": "Standard Delivery Query (Bengali)", "query": "ডেলিভারি চার্জ কত?"},
    {"label": "Standard Delivery Query (English)", "query": "What is the shipping cost inside Dhaka?"},
    {"label": "Colloquial Return/Refund Query", "query": "মাল ফেরত দিবার নিয়ম কি?"},
    {"label": "Slang Payment Query", "query": "টাকা পেমেন্ট করব কেমনে বিকাশ আছে?"},
    {"label": "Out of Domain Query", "query": "আজকের আবহাওয়া কেমন?"},
]

print("=" * 110)
print(f"{'QUERY':<35} | {'TIER EXECUTED':<22} | {'SCORE':<7} | {'TIME(ms)':<8} | {'TOP RESULT'}")
print("=" * 110)

for item in queries:
    payload = json.dumps({
        "query": item["query"],
        "workspace_id": 1,
        "top_k": 3
    }).encode("utf-8")
    
    req = urllib.request.Request(API_URL, data=payload, headers={"Content-Type": "application/json"})
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as resp:
            elapsed = (time.time() - t0) * 1000
            data = json.loads(resp.read().decode("utf-8"))
            telemetry = data.get("telemetry") or {}
            tier = telemetry.get("tier_executed", "N/A")
            results = data.get("results", [])
            if results:
                top = results[0]
                q_text = top.get("question", "")[:40]
                score = round(top.get("score", 0.0), 3)
            else:
                q_text = "NONE (No match)"
                score = 0.0
            
            print(f"{item['query']:<35} | {tier:<22} | {score:<7} | {elapsed:<8.1f} | {q_text}")
    except Exception as e:
        print(f"{item['query']:<35} | ERROR: {e}")

print("=" * 110)
