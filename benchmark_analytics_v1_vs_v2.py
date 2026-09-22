import asyncio
import logging
import os
from dotenv import load_dotenv

load_dotenv()

from app.analytics.planner import AnalyticsPlanner
from app.analytics.compiler import AnalyticsCompiler
from app.analytics.planner_v2 import SemanticPlannerV2
from app.analytics.compiler_v2 import AnalyticsCompilerV2
from app.analytics.executor import AnalyticsExecutor

# Setup logging
logging.basicConfig(level=logging.ERROR)

planner_v1 = AnalyticsPlanner()
compiler_v1 = AnalyticsCompiler()

planner_v2 = SemanticPlannerV2()
compiler_v2 = AnalyticsCompilerV2()
executor = AnalyticsExecutor()

workspace_id = 1

test_queries = [
    "total sales this month",
    "month wise sale data daw?",
    "what are the top 3 selling products?",
    "koyta order pending ache?",
    "average order value koto?",
    "show me sales from august",
]

def run_comparison():
    print("=========================================================")
    print("    ANALYTICS V1 vs V2 SHADOW MODE COMPARISON BENCHMARK   ")
    print("=========================================================\n")
    
    matches = 0
    total = len(test_queries)
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test [{i}/{total}]: \"{query}\"")
        
        # --- Run V1 (Baseline) ---
        try:
            plan_v1, _ = planner_v1.plan(query)
            sql_v1, params_v1 = compiler_v1.compile(plan_v1, workspace_id)
            rows_v1, _ = executor.execute(sql_v1, params_v1)
            v1_status = "SUCCESS"
        except Exception as e:
            v1_status = f"FAILED ({str(e)})"
            rows_v1 = []
            
        # --- Run V2 (Semantic) ---
        try:
            plan_v2, _ = planner_v2.plan(query)
            sql_v2, params_v2 = compiler_v2.compile(plan_v2, workspace_id)
            rows_v2, _ = executor.execute(sql_v2, params_v2)
            v2_status = "SUCCESS"
        except Exception as e:
            v2_status = f"FAILED ({str(e)})"
            rows_v2 = []
            
        # --- Compare Outputs ---
        is_match = (rows_v1 == rows_v2) and v1_status == "SUCCESS" and v2_status == "SUCCESS"
        if is_match:
            matches += 1
            print(" ✅ MATCH")
        else:
            print(" ❌ MISMATCH")
            print(f"    V1 Status: {v1_status}")
            print(f"    V2 Status: {v2_status}")
            if v1_status == "SUCCESS" and v2_status == "SUCCESS":
                print(f"    V1 SQL: {sql_v1}")
                print(f"    V2 SQL: {sql_v2}")
                print(f"    V1 Rows: {len(rows_v1)}")
                print(f"    V2 Rows: {len(rows_v2)}")
        print("-" * 50)
        
    print(f"\nFinal Score: {matches}/{total} ({round(matches/total*100, 2)}% Confidence)")
    
if __name__ == "__main__":
    run_comparison()
