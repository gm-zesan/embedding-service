import json
import urllib.request
import time

API_URL = "http://127.0.0.1:8001/api/v1/search"

banglish_queries = [
    {
        "category": "Delivery & Shipping",
        "query": "delivery charge koto?",
        "expected": "Delivery & Shipping Rates",
    },
    {
        "category": "Delivery Details",
        "query": "dhakar vitore delivery fee koto ar koydin lagbe?",
        "expected": "Delivery & Shipping Rates",
    },
    {
        "category": "Return Policy",
        "query": "product return korar rules ki?",
        "expected": "Official Return Policy",
    },
    {
        "category": "Payment Method",
        "query": "bkash diye payment kora jabe ki?",
        "expected": "Accepted Payment Methods",
    },
    {
        "category": "Refund Process",
        "query": "taka refund pabo kivabe?",
        "expected": "Official Return Policy / Refund",
    },
    {
        "category": "WhatsApp Connect",
        "query": "whatsapp kivabe connect korbo?",
        "expected": "How do I connect WhatsApp?",
    },
    {
        "category": "Account Creation",
        "query": "notun account khulte ki ki lage?",
        "expected": "How do I create an account?",
    },
    {
        "category": "Out of Domain",
        "query": "ajker weather kemon?",
        "expected": "REJECTED (OOD)",
    },
]

print("=" * 120)
print(f"{'BANGLISH QUERY':<45} | {'TIER EXECUTED':<22} | {'SCORE':<7} | {'TIME(ms)':<8} | {'RETRIEVED FAQ'}")
print("=" * 120)

for item in banglish_queries:
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
            
            print(f"{item['query']:<45} | {tier:<22} | {score:<7} | {elapsed:<8.1f} | {q_text}")
    except Exception as e:
        print(f"{item['query']:<45} | ERROR: {e}")

print("=" * 120)
