import re

with open('benchmark_100_queries.py', 'r') as f:
    content = f.read()

content = content.replace("os.environ.pop('http_proxy', None)\n", "")
content = content.replace("os.environ.pop('https_proxy', None)\n", "")
content = content.replace("os.environ.pop('HTTP_PROXY', None)\n", "")
content = content.replace("os.environ.pop('HTTPS_PROXY', None)\n", "")

with open('benchmark_100_queries.py', 'w') as f:
    f.write(content)
