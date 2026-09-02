import re

with open('benchmark_100_queries.py', 'r') as f:
    content = f.read()

# Replace the httpx POST with direct function call
new_content = content.replace(
    'async with httpx.AsyncClient(timeout=25.0) as client:',
    'from app.retrieval_engine import search_knowledge_base\n    async with httpx.AsyncClient(timeout=25.0) as client:'
)

new_content = re.sub(
    r'resp = await client\.post\([^)]+\)\s*duration_ms = round\(\(time\.time\(\) - t0\) \* 1000, 1\)',
    'data = await search_knowledge_base(query, workspace_id=0, top_k=5)\n                duration_ms = round((time.time() - t0) * 1000, 1)',
    new_content,
    flags=re.MULTILINE
)

new_content = re.sub(
    r'if resp\.status_code != 200:\s*print\(f"\[\{idx\}\] HTTP ERROR \{resp\.status_code\}: \{query\}"\)\s*continue\s*data = resp\.json\(\)',
    '',
    new_content,
    flags=re.MULTILINE
)

with open('benchmark_100_queries.py', 'w') as f:
    f.write(new_content)
