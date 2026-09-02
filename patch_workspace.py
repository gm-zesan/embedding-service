import re

with open('benchmark_100_queries.py', 'r') as f:
    content = f.read()

content = content.replace("workspace_id=0", "workspace_id=1")

with open('benchmark_100_queries.py', 'w') as f:
    f.write(content)
